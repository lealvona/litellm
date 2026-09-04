import re

import pytest

import privacy_guard


@pytest.fixture
def numeric_rules(monkeypatch):
    monkeypatch.setattr(
        privacy_guard._RULES,
        "regex",
        [
            (
                "email",
                re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
                False,
            ),
            (
                "credit_card",
                re.compile(r"\b(?:\d[ -]?){13,19}\b"),
                True,
            ),
            (
                "phone",
                re.compile(r"\b(?:\+?\d{1,2}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"),
                False,
            ),
        ],
    )
    monkeypatch.setattr(privacy_guard._RULES, "phrase", [])
    monkeypatch.setattr(privacy_guard._RULES, "keyword", [])
    monkeypatch.setattr(privacy_guard._RULES, "load", lambda force=False: None)
    monkeypatch.setattr(privacy_guard._RULES, "raw", {"enabled": True})


def test_unformatted_tool_timestamp_is_not_a_phone(numeric_rules):
    messages = [
        {
            "role": "tool",
            "content": '{"timestamp": 1786976605.5180418, "status": "ok"}',
        }
    ]

    assert privacy_guard.evaluate(messages) == (False, None)


def test_openssh_capability_identifier_is_not_an_email(numeric_rules):
    messages = [
        {
            "role": "tool",
            "content": "server option security-capability@openssh.com is enabled",
        }
    ]

    assert privacy_guard.evaluate(messages) == (False, None)


def test_public_email_in_tool_output_does_not_force_local(numeric_rules):
    messages = [{"role": "tool", "content": "Maintainer: person@example.com"}]

    assert privacy_guard.evaluate(messages) == (False, None)


def test_user_authored_email_remains_sensitive(numeric_rules):
    messages = [{"role": "user", "content": "My address is person@example.com"}]

    assert privacy_guard.evaluate(messages) == (True, "email")


def test_personal_mail_intent_protects_tool_email_output(numeric_rules, monkeypatch):
    monkeypatch.setattr(
        privacy_guard._RULES,
        "phrase",
        [
            (
                "personal_mail",
                re.compile(r"\b(?:my\s+inbox|read\s+my\s+email)\b", re.IGNORECASE),
            )
        ],
    )
    messages = [
        {"role": "user", "content": "Read my inbox and summarize it"},
        {"role": "tool", "content": "From: person@example.com"},
    ]

    assert privacy_guard.evaluate(messages) == (True, "personal_mail")


def test_tool_secret_path_reference_is_not_secret_content(numeric_rules, monkeypatch):
    monkeypatch.setattr(
        privacy_guard._RULES,
        "phrase",
        [
            (
                "secret_assignment",
                re.compile(
                    r"\b(?:password|secret|api[_ ]?key|private[_ ]?key)\b\s*(?:is|=|:)\s*\S",
                    re.IGNORECASE,
                ),
            )
        ],
    )
    messages = [{"role": "tool", "content": "private key: /home/user/.ssh/id_ed25519"}]

    assert privacy_guard.evaluate(messages) == (False, None)


def test_tool_secret_environment_reference_is_not_secret_content(numeric_rules, monkeypatch):
    monkeypatch.setattr(
        privacy_guard._RULES,
        "phrase",
        [
            (
                "secret_assignment",
                re.compile(
                    r"\b(?:password|secret|api[_ ]?key|private[_ ]?key)\b\s*(?:is|=|:)\s*\S",
                    re.IGNORECASE,
                ),
            )
        ],
    )
    messages = [{"role": "tool", "content": "api_key: os.environ/PROVIDER_API_KEY"}]

    assert privacy_guard.evaluate(messages) == (False, None)


def test_tool_secret_template_reference_is_not_secret_content(numeric_rules, monkeypatch):
    monkeypatch.setattr(
        privacy_guard._RULES,
        "phrase",
        [
            (
                "secret_assignment",
                re.compile(
                    r"\b(?:password|secret|api[_ ]?key|private[_ ]?key)\b\s*(?:is|=|:)\s*\S",
                    re.IGNORECASE,
                ),
            )
        ],
    )
    messages = [{"role": "tool", "content": r"private key: {SSH_KEY}\\id_ed25519"}]

    assert privacy_guard.evaluate(messages) == (False, None)


def test_actual_tool_secret_assignment_remains_sensitive(numeric_rules, monkeypatch):
    monkeypatch.setattr(
        privacy_guard._RULES,
        "phrase",
        [
            (
                "secret_assignment",
                re.compile(
                    r"\b(?:password|secret|api[_ ]?key|private[_ ]?key)\b\s*(?:is|=|:)\s*\S",
                    re.IGNORECASE,
                ),
            )
        ],
    )
    messages = [{"role": "tool", "content": "password: correct-horse-battery-staple"}]

    assert privacy_guard.evaluate(messages) == (True, "secret_assignment")


def test_unformatted_user_phone_remains_sensitive(numeric_rules):
    messages = [{"role": "user", "content": "Call me at 2125550198"}]

    assert privacy_guard.evaluate(messages) == (True, "phone")


def test_formatted_phone_in_tool_output_remains_sensitive(numeric_rules):
    messages = [{"role": "tool", "content": "customer phone: 212-555-0198"}]

    assert privacy_guard.evaluate(messages) == (True, "phone")


def test_luhn_valid_date_stamped_id_is_not_a_credit_card(numeric_rules):
    # This exact shape appears in generated session filenames. Some values pass
    # Luhn by chance and must not permanently pin a technical session local.
    messages = [{"role": "tool", "content": "session_20260523-234834.json"}]

    assert privacy_guard.evaluate(messages) == (False, None)


def test_real_credit_card_candidate_remains_sensitive(numeric_rules):
    messages = [{"role": "tool", "content": "payment card 4111 1111 1111 1111"}]

    assert privacy_guard.evaluate(messages) == (True, "credit_card")


@pytest.mark.asyncio
async def test_context_mode_does_not_honor_old_permanent_fuse(monkeypatch):
    monkeypatch.setattr(privacy_guard._RULES, "load", lambda force=False: None)
    monkeypatch.setattr(
        privacy_guard._RULES,
        "raw",
        {
            "enabled": True,
            "router_name": "smart-router",
            "local_model": "llm_115",
            "fuse_mode": "context",
        },
    )
    monkeypatch.setattr(privacy_guard, "evaluate", lambda messages: (False, None))
    data = {
        "model": "smart-router",
        "user": "previously-fused-session",
        "messages": [{"role": "user", "content": "Reason carefully about this design."}],
    }
    privacy_guard._FUSED.add("sid:previously-fused-session")

    result = await privacy_guard.proxy_handler_instance.async_pre_call_hook(
        None, None, data, "acompletion"
    )

    assert result["model"] == "smart-router"


@pytest.mark.asyncio
async def test_context_mode_still_routes_current_sensitive_payload_local(monkeypatch):
    monkeypatch.setattr(privacy_guard._RULES, "load", lambda force=False: None)
    monkeypatch.setattr(
        privacy_guard._RULES,
        "raw",
        {
            "enabled": True,
            "router_name": "smart-router",
            "local_model": "llm_115",
            "fuse_mode": "context",
        },
    )
    monkeypatch.setattr(privacy_guard, "evaluate", lambda messages: (True, "private_key_block"))
    monkeypatch.setattr(privacy_guard, "_log_decision", lambda *args, **kwargs: None)
    data = {
        "model": "smart-router",
        "user": "active-sensitive-session",
        "messages": [{"role": "user", "content": "sensitive placeholder"}],
    }

    result = await privacy_guard.proxy_handler_instance.async_pre_call_hook(
        None, None, data, "acompletion"
    )

    assert result["model"] == "llm_115"


# ── Context-compaction routing ───────────────────────────────────────────────
# A compaction prompt is one machine-built user message. These tests pin the
# two properties the whole design rests on: the fence restores per-turn roles
# (so a compaction is graded exactly like the conversation it summarizes), and
# anything unfenced falls back to a whole-payload scan, which routes local.


def _fence(text: str, kind: str = "transcript") -> str:
    return (
        f"{privacy_guard.MATERIAL_BEGIN}: {kind} ===\n"
        f"{text}\n{privacy_guard.MATERIAL_END}"
    )


_TRANSCRIPT = """[USER]: check the rom scraper on the arcade box
[ASSISTANT]: looking now
[Tool calls:
  terminal(systemctl status scraper)
]

[TOOL RESULT call_9f2]: {"maintainer": "person@example.com", "build": 1786976605}

[ASSISTANT]: it is running"""


def test_fence_restores_turn_roles():
    messages = privacy_guard.compaction_messages(_fence(_TRANSCRIPT))

    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]


def test_unfenced_payload_is_not_treated_as_a_compaction():
    assert privacy_guard.compaction_messages("no fence here") is None


def test_multiple_fenced_blocks_are_all_scanned():
    payload = (
        "PREVIOUS SUMMARY:\n"
        + _fence("Earlier work covered the scraper.")
        + "\n\nNEW TURNS TO INCORPORATE:\n"
        + _fence(_TRANSCRIPT)
    )

    messages = privacy_guard.compaction_messages(payload)

    assert len(messages) == 5
    assert messages[0] == {"role": "user", "content": "Earlier work covered the scraper."}


def test_summary_block_grades_as_machine_written():
    """A summary is scrubbed before it is written, so it grades leniently.

    Graded strictly, a ten-digit build id carried forward by the summary would
    read as a phone number and pin every later compaction to the local model.
    """
    messages = privacy_guard.compaction_messages(
        _fence("Checked build 1786976605 on scraper.service", kind="summary")
    )

    assert [m["role"] for m in messages] == ["tool"]


def test_memory_block_still_grades_strictly():
    messages = privacy_guard.compaction_messages(
        _fence("Contact preference recorded.", kind="memory")
    )

    assert [m["role"] for m in messages] == ["user"]


def test_summary_block_does_not_excuse_a_real_address(numeric_rules):
    """Lenient does not mean unscanned: the personal-mail shapes still trip."""
    payload = _fence("Queued the report to hannah@example.com", kind="summary")

    # Bare addresses in machine output are exempt by design (see the email
    # rule), so this asserts the block is scanned at all, not that it trips.
    assert privacy_guard.compaction_messages(payload)[0]["role"] == "tool"
    assert privacy_guard.evaluate(
        privacy_guard.compaction_messages(_fence("[USER]: hannah@example.com"))
    ) == (True, "email")


def test_unlabelled_block_is_graded_strictly(numeric_rules):
    """A previous summary or memory section carries no role labels.

    It is derived from the conversation, so it must be graded as user-authored
    rather than inheriting the lenient tool-output path.
    """
    payload = _fence("Prior work: reached the vendor at contact@example.com.")

    assert privacy_guard.evaluate(privacy_guard.compaction_messages(payload)) == (
        True,
        "email",
    )


def test_tool_output_alone_does_not_force_compaction_local(numeric_rules):
    """The point of the fence: without it every compaction would route local.

    Flattened, the public maintainer address and the epoch build id read as
    user-authored and trip the guard. With roles restored they do not.
    """
    payload = _fence(_TRANSCRIPT)

    assert privacy_guard.evaluate([{"role": "user", "content": payload}]) == (True, "email")
    assert privacy_guard.evaluate(privacy_guard.compaction_messages(payload)) == (False, None)


def test_user_authored_pii_still_routes_compaction_local(numeric_rules):
    payload = _fence("[USER]: mail the invoice to hannah@example.com\n[ASSISTANT]: ok")

    assert privacy_guard.evaluate(privacy_guard.compaction_messages(payload)) == (
        True,
        "email",
    )


@pytest.fixture
def compaction_rules(numeric_rules, monkeypatch, tmp_path):
    # Never append to the operator's live decision feed: the admin UI reads it.
    monkeypatch.setattr(
        privacy_guard, "DECISIONS_LOG", str(tmp_path / "decisions.jsonl")
    )
    monkeypatch.setattr(
        privacy_guard._RULES,
        "raw",
        {
            "enabled": True,
            "router_name": "smart-router",
            "local_model": "llm_115",
            "fuse_mode": "context",
            "compression_router": "compression-router",
            "compression_cloud_model": "minimax-m3",
        },
    )


def _run_hook(data):
    import asyncio

    return asyncio.run(
        privacy_guard.proxy_handler_instance.async_pre_call_hook(
            None, None, data, "completion"
        )
    )


def test_clean_compaction_is_promoted_to_the_cloud_summarizer(compaction_rules):
    data = {
        "model": "compression-router",
        "messages": [{"role": "user", "content": _fence(_TRANSCRIPT)}],
    }

    assert _run_hook(data)["model"] == "minimax-m3"


def test_sensitive_compaction_stays_local(compaction_rules):
    data = {
        "model": "compression-router",
        "messages": [
            {
                "role": "user",
                "content": _fence("[USER]: mail the invoice to hannah@example.com"),
            }
        ],
    }

    assert _run_hook(data)["model"] == "llm_115"


def test_unfenced_compaction_stays_local(compaction_rules):
    """No fence means no role information, so the strict scan decides."""
    data = {
        "model": "compression-router",
        "messages": [{"role": "user", "content": _TRANSCRIPT}],
    }

    assert _run_hook(data)["model"] == "llm_115"


def test_permanent_fuse_never_promotes_a_compaction(compaction_rules, monkeypatch):
    raw = dict(privacy_guard._RULES.raw)
    raw["fuse_mode"] = "permanent"
    monkeypatch.setattr(privacy_guard._RULES, "raw", raw)
    data = {
        "model": "compression-router",
        "messages": [{"role": "user", "content": _fence(_TRANSCRIPT)}],
    }

    assert _run_hook(data)["model"] == "llm_115"


def test_no_cloud_summarizer_configured_stays_local(compaction_rules, monkeypatch):
    raw = dict(privacy_guard._RULES.raw)
    raw["compression_cloud_model"] = ""
    monkeypatch.setattr(privacy_guard._RULES, "raw", raw)
    data = {
        "model": "compression-router",
        "messages": [{"role": "user", "content": _fence(_TRANSCRIPT)}],
    }

    assert _run_hook(data)["model"] == "llm_115"


def test_other_models_are_untouched(compaction_rules):
    data = {"model": "glm-5.3", "messages": [{"role": "user", "content": "hello"}]}

    assert _run_hook(data)["model"] == "glm-5.3"


def test_router_names_list_supersedes_router_name(monkeypatch):
    monkeypatch.setattr(
        privacy_guard._RULES,
        "raw",
        {"router_name": "smart-router", "router_names": ["smart-router", "router-b"]},
    )

    assert privacy_guard._RULES.router_names == ("smart-router", "router-b")


def test_router_name_alone_still_works(monkeypatch):
    monkeypatch.setattr(privacy_guard._RULES, "raw", {"router_name": "smart-router"})

    assert privacy_guard._RULES.router_names == ("smart-router",)


def test_assistant_build_id_does_not_pin_a_compaction_local(numeric_rules):
    """A ten-digit build id quoted by the assistant is not a phone number.

    Graded as human-authored it matched the phone rule, which pinned every
    compaction of a long technical conversation to the local model.
    """
    payload = _fence(
        "[USER]: check the scraper\n"
        "[ASSISTANT]: scraper.service is active; build 1786976605"
    )
    messages = privacy_guard.compaction_messages(payload)

    assert privacy_guard.evaluate(messages) == (True, "phone")  # default grading
    assert privacy_guard.evaluate(
        messages, privacy_guard.TRANSCRIPT_MACHINE_ROLES
    ) == (False, None)


def test_machine_role_leniency_does_not_reach_the_user_turn(numeric_rules):
    """The exemption is for machine text; the human turn stays protected."""
    payload = _fence("[USER]: text me at 555-867-5309\n[ASSISTANT]: noted")

    assert privacy_guard.evaluate(
        privacy_guard.compaction_messages(payload),
        privacy_guard.TRANSCRIPT_MACHINE_ROLES,
    ) == (True, "phone")


def test_ordinary_requests_keep_the_strict_default(numeric_rules):
    """Nothing about a normal turn changes: only ``tool`` is machine text."""
    messages = [{"role": "assistant", "content": "reference 1786976605"}]

    assert privacy_guard.evaluate(messages) == (True, "phone")
