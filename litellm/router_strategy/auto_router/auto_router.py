"""
Auto-Routing Strategy that works with a Semantic Router Config
"""

import asyncio
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger

# Hard wall-clock budget for the semantic-routing embedding (init + query),
# lealvona 2026-07-17. Healthy is ~27ms/query; a dead embedder must degrade to
# the default model within this, never wedge the router. Env-tunable.
_AUTOROUTER_TIMEOUT = float(os.environ.get("LITELLM_AUTOROUTER_TIMEOUT", "6.0"))

if TYPE_CHECKING:
    from semantic_router.routers.base import Route

    from litellm.router import Router
    from litellm.types.router import PreRoutingHookResponse
else:
    Router = Any
    PreRoutingHookResponse = Any
    Route = Any


class AutoRouter(CustomLogger):
    DEFAULT_AUTO_SYNC_VALUE = "local"

    def __init__(
        self,
        model_name: str,
        default_model: str,
        embedding_model: str,
        litellm_router_instance: "Router",
        auto_router_config_path: Optional[str] = None,
        auto_router_config: Optional[str] = None,
    ):
        """
        Auto-Router class that uses a semantic router to route requests to the appropriate model.

        Args:
            model_name: The name of the model to use for the auto-router. eg. if model = "auto-router1" then us this router.
            auto_router_config_path: The path to the router config file.
            auto_router_config: The config to use for the auto-router. You can either use this or auto_router_config_path, not both.
            default_model: The default model to use if no route is found.
            embedding_model: The embedding model to use for the auto-router.
            litellm_router_instance: The instance of the LiteLLM Router.
        """
        from semantic_router.routers import SemanticRouter

        self.auto_router_config_path: Optional[str] = auto_router_config_path
        self.auto_router_config: Optional[str] = auto_router_config
        self.auto_sync_value = self.DEFAULT_AUTO_SYNC_VALUE
        self.loaded_routes: List[Route] = self._load_semantic_routing_routes()
        self.routelayer: Optional[SemanticRouter] = None
        self.default_model = default_model
        self.embedding_model: str = embedding_model
        self.litellm_router_instance: "Router" = litellm_router_instance

    def _load_semantic_routing_routes(self) -> List[Route]:
        from semantic_router.routers import SemanticRouter

        if self.auto_router_config_path:
            return SemanticRouter.from_json(self.auto_router_config_path).routes
        elif self.auto_router_config:
            return self._load_auto_router_routes_from_config_json()
        else:
            raise ValueError("No router config provided")

    def _load_auto_router_routes_from_config_json(self) -> List[Route]:
        import json

        from semantic_router.routers.base import Route

        if self.auto_router_config is None:
            raise ValueError("No auto router config provided")
        auto_router_routes: List[Route] = []
        loaded_config = json.loads(self.auto_router_config)
        for route in loaded_config.get("routes", []):
            auto_router_routes.append(
                Route(
                    name=route.get("name"),
                    description=route.get("description"),
                    utterances=route.get("utterances", []),
                    score_threshold=route.get("score_threshold"),
                )
            )
        return auto_router_routes

    @staticmethod
    def _extract_text_from_messages(messages: List[Dict[str, Any]]) -> str:
        """
        Extract text content from the last user message for routing.

        Handles tool-call conversations (where the last message may be an
        assistant or tool message with non-string content) and multimodal
        messages (where content is a list of content blocks).
        """
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if content is None:
                    return ""
                if isinstance(content, list):
                    return " ".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                return str(content)
        return ""

    def _route_sync(self, text: str):
        """Synchronous routing: lazily builds the SemanticRouter (embeds every
        route utterance on first use) then embeds ``text`` and returns the route
        choice. Both are blocking embedding calls, so this is always invoked via
        asyncio.to_thread + asyncio.wait_for (see async_pre_routing_hook) — that
        keeps the proxy event loop responsive and bounds a hung embedder so
        routing degrades to the default model instead of wedging the fleet."""
        from semantic_router.routers import SemanticRouter

        from litellm.router_strategy.auto_router.litellm_encoder import (
            LiteLLMRouterEncoder,
        )

        if self.routelayer is None:
            self.routelayer = SemanticRouter(
                routes=self.loaded_routes,
                encoder=LiteLLMRouterEncoder(
                    litellm_router_instance=self.litellm_router_instance,
                    model_name=self.embedding_model,
                ),
                auto_sync=self.auto_sync_value,
            )
        return self.routelayer(text=text)

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Dict,
        messages: Optional[List[Dict[str, Any]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ) -> Optional["PreRoutingHookResponse"]:
        """
        This hook is called before the routing decision is made.

        Used for the litellm auto-router to modify the request before the routing decision is made.
        """
        from semantic_router.routers import SemanticRouter
        from semantic_router.schema import RouteChoice

        from litellm.router_strategy.auto_router.litellm_encoder import (
            LiteLLMRouterEncoder,
        )
        from litellm.types.router import PreRoutingHookResponse

        if messages is None:
            # do nothing, return same inputs
            return None

        # Route the request under a hard wall-clock bound, OFF the event loop.
        # Both SemanticRouter init (embeds every route utterance) and __call__
        # (embeds the query) are SYNCHRONOUS embedding calls; running them inline
        # blocks the ENTIRE proxy event loop, so a dead/slow embedder wedged
        # every routed request for 50-125s (2026-07-15/17). to_thread frees the
        # loop; wait_for guarantees we fall through to the default model within
        # _AUTOROUTER_TIMEOUT instead of hanging.
        model = self.default_model
        route_choice: Optional[Union[RouteChoice, List[RouteChoice]]] = None
        try:
            message_content = self._extract_text_from_messages(messages)
            route_choice = await asyncio.wait_for(
                asyncio.to_thread(self._route_sync, message_content),
                timeout=_AUTOROUTER_TIMEOUT,
            )
        except Exception as e:
            verbose_router_logger.warning(
                f"auto_router: semantic routing failed/timed out ({e}); "
                f"using default model {self.default_model}"
            )
        verbose_router_logger.debug(f"route_choice: {route_choice}")
        if isinstance(route_choice, RouteChoice):
            model = route_choice.name or self.default_model
        elif isinstance(route_choice, list) and route_choice:
            model = route_choice[0].name or self.default_model

        # --- thinking-toggle injection (Phase B, lealvona 2026-06-23) ------
        # Translate any reasoning_effort signal on the request into the CHOSEN
        # model's native thinking param via extra_body, POST-routing.
        # request_kwargs is the live completion-call dict (same object used
        # downstream at router.py async_get_healthy_deployments / acompletion;
        # cf. QualityRouter which stashes its decision here and reads it
        # post-response), so extra_body injected here reaches the deployment
        # call and survives the proxy's drop_params:true.
        # Best-effort: an import/translation failure must NEVER break routing.
        try:
            try:
                import thinking_control as _tc
            except ImportError:
                # Out-of-tree patch module; when the proxy runs from an installed
                # copy (not the repo), point LITELLM_PATCH_DIR at the checkout that
                # holds thinking_control.py. No path is hardcoded here.
                _p = os.environ.get("LITELLM_PATCH_DIR")
                if _p:
                    import sys as _sys
                    if _p not in _sys.path:
                        _sys.path.insert(0, _p)
                import thinking_control as _tc
            if isinstance(request_kwargs, dict):
                _st, _ef = _tc._norm(request_kwargs.get("reasoning_effort"))
                if _st is not None:
                    _tc.apply_thinking(request_kwargs, _st, _ef, model=model)
        except Exception:
            pass
        # ------------------------------------------------------------------

        return PreRoutingHookResponse(
            model=model,
            messages=messages,
        )
