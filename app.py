"""
LearnPath AI
Pick a topic -> best free courses + notes + labs -> AI roadmap -> quiz -> badge -> AI assistant.
Run:  streamlit run app.py
"""
import copy
import os
from datetime import date

import streamlit as st

from badge import make_badge_png
from courses import FORMATS, LEVELS, load_catalog, max_score, rank_courses
from llm import (
    LLM,
    PROVIDERS,
    LLMError,
    assistant_reply,
    explain_pick,
    generate_notes,
    generate_quiz,
    generate_roadmap,
)

PASS_MARK = 70   # quiz score (%) needed for the badge
QUIZ_SIZE = 8

st.set_page_config(page_title="LearnPath AI", page_icon="🎓", layout="wide")

# ---------------------------------------------------------------- state
DEFAULTS = {
    "roadmap": None,       # dict from generate_roadmap
    "roadmap_for": None,   # (topic, level) the roadmap was made for
    "rid": 0,              # bumps on reset so old checkbox values are dropped
    "notes": {},           # module index -> markdown
    "quiz": None,
    "qid": 0,              # bumps for every new quiz so old answers are dropped
    "results": [],
    "last_score": None,
    "best_score": 0,
    "chat": [],
    "picked": None,        # {"title": ..., "topic": ...}
    "pick_text": None,     # {"topic": ..., "text": ...}
    "celebrated": False,
}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, copy.deepcopy(_v))


def reset_learning_state():
    """Clear roadmap, progress, quiz and chat (keeps settings and the picked course)."""
    keep_rid = st.session_state.rid + 1
    keep_qid = st.session_state.qid + 1
    for key in ("roadmap", "roadmap_for", "notes", "quiz", "results", "last_score",
                "best_score", "chat", "celebrated"):
        st.session_state[key] = copy.deepcopy(DEFAULTS[key])
    st.session_state.rid = keep_rid
    st.session_state.qid = keep_qid


def get_secret(name: str) -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, "")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    user_name = st.text_input("Your name (printed on the badge)", key="user_name")
    provider = st.selectbox("AI provider", list(PROVIDERS))
    model_label = st.selectbox("Model", list(PROVIDERS[provider]["models"]))
    key_name = PROVIDERS[provider]["key_name"]
    api_key = get_secret(key_name)
    if api_key:
        st.success(f"{key_name} found")
    else:
        api_key = st.text_input(key_name, type="password", help="Used only in this session. Never saved.")
    st.caption("Your key stays in this browser session. It is not written to disk or GitHub.")
    st.divider()
    if st.button("🔄 Start over (clear roadmap, quiz, chat)"):
        reset_learning_state()
        st.session_state.picked = None
        st.rerun()

llm = LLM(provider, PROVIDERS[provider]["models"][model_label], api_key) if api_key else None


def need_llm() -> bool:
    if llm is None:
        st.warning("Add your API key in the sidebar first.")
        return False
    return True


def run_ai(fn, *args):
    """Call an AI task with a spinner. Shows a friendly error and returns None on failure."""
    try:
        with st.spinner("AI is working..."):
            return fn(llm, *args)
    except LLMError as exc:
        st.error(str(exc))
        return None


# ---------------------------------------------------------------- page
catalog = load_catalog()
st.title("🎓 LearnPath AI")
st.caption("Choose a topic → get the best free course, notes and labs → follow a roadmap → pass the quiz → earn your badge.")

tab_pick, tab_road, tab_quiz, tab_badge, tab_chat = st.tabs(
    ["1️⃣ Pick a course", "2️⃣ Roadmap", "3️⃣ Quiz", "4️⃣ Progress & badge", "💬 AI assistant"]
)

# ================================================================ 1. PICK
with tab_pick:
    c1, c2, c3, c4 = st.columns(4)
    topic = c1.selectbox("Topic you are interested in", list(catalog), key="topic")
    level = c2.selectbox("Your level", LEVELS, key="level")
    fmt = c3.selectbox("Learning style", FORMATS, key="fmt")
    free_only = c4.checkbox("Free courses only", value=True, key="free_only")
    st.write(catalog[topic]["description"])

    ranked = rank_courses(catalog[topic]["courses"], level, fmt, free_only)
    if not ranked:
        st.info("No course matches these choices. Try another level or untick 'Free only'.")
    for idx, (score, c) in enumerate(ranked):
        with st.container(border=True):
            st.markdown(f"### {'🏆 ' if idx == 0 else ''}[{c['title']}]({c['url']})")
            practice = "🧪 has practice" if c.get("has_labs") else "no built-in practice"
            st.caption(f"{c['platform']} • {c['level']} • {c['price']} • {c['format']} • ~{c['hours']}h (approx.) • {practice}")
            st.write(c["why"])
            st.progress(min(score / max_score(fmt), 1.0), text="Fit with your choices")
            picked = st.session_state.picked
            if picked and picked["topic"] == topic and picked["title"] == c["title"]:
                st.success("✅ This is your chosen course. Continue to step 2 (Roadmap).")
            elif st.button("Choose this course", key=f"pick_{topic}_{idx}"):
                st.session_state.picked = {"title": c["title"], "topic": topic}
                st.rerun()

    if ranked and st.button("🤖 Ask AI which one is best for me"):
        if need_llm():
            text = run_ai(explain_pick, topic, level, [c for _, c in ranked[:3]])
            if text:
                st.session_state.pick_text = {"topic": topic, "text": text}
    pt = st.session_state.pick_text
    if pt and pt["topic"] == topic:
        st.info(pt["text"])

    st.divider()
    n_col, l_col = st.columns(2)
    with n_col:
        st.subheader("📝 Notes & cheat sheets")
        for n in catalog[topic]["notes"]:
            st.markdown(f"- [{n['title']}]({n['url']}) - {n['note']}")
    with l_col:
        st.subheader("🧪 Practice labs")
        for lab in catalog[topic]["labs"]:
            st.markdown(f"- [{lab['title']}]({lab['url']}) - {lab['note']}")
    st.caption(
        "No platform tags notes as 'handwritten', so this app does not pretend to find them. "
        "These are good free written notes. In step 2 the AI can also write short notes for every module. "
        "Only practice security skills on legal labs like these."
    )

# ================================================================ 2. ROADMAP
rm = st.session_state.roadmap
with tab_road:
    st.write("The AI builds a step-by-step plan for your topic and level.")
    if rm and st.session_state.roadmap_for != (topic, level):
        old_topic, old_level = st.session_state.roadmap_for
        st.info(f"Your current roadmap is for **{old_topic} ({old_level})**. "
                "Making a new one clears your progress, quiz and chat.")
    if st.button(f"🗺️ Generate roadmap for {topic} ({level})", type="primary"):
        if need_llm():
            data = run_ai(generate_roadmap, topic, level)
            if data:
                reset_learning_state()
                st.session_state.roadmap = data
                st.session_state.roadmap_for = (topic, level)
                st.rerun()

    if rm:
        rm_topic, rm_level = st.session_state.roadmap_for
        rid = st.session_state.rid
        modules = rm["modules"]
        st.subheader(rm["title"])
        st.write(rm["summary"])
        done = sum(1 for i in range(len(modules)) if st.session_state.get(f"mod_{rid}_{i}", False))
        st.progress(done / len(modules), text=f"{done} of {len(modules)} modules finished")

        for i, m in enumerate(modules):
            with st.expander(f"Module {i + 1}: {m['title']}  •  ~{m['hours']}h", expanded=(i == 0)):
                st.markdown(f"**Goal:** {m['goal']}")
                if m["topics"]:
                    st.markdown("**You will learn:**\n" + "\n".join(f"- {t}" for t in m["topics"]))
                st.markdown(f"**Practice:** {m['practice']}")
                st.checkbox("✅ I finished this module", key=f"mod_{rid}_{i}")
                if st.button("📝 Write study notes for this module", key=f"notebtn_{rid}_{i}"):
                    if need_llm():
                        text = run_ai(generate_notes, rm_topic, rm_level, m)
                        if text:
                            st.session_state.notes[i] = text
                if i in st.session_state.notes:
                    st.markdown(st.session_state.notes[i])
                    st.download_button("⬇️ Download notes (.md)", st.session_state.notes[i],
                                       file_name=f"{rm_topic}-module-{i + 1}-notes.md".replace(" ", "-"),
                                       key=f"dl_{rid}_{i}")

# ================================================================ 3. QUIZ
with tab_quiz:
    if not rm:
        st.info("Make your roadmap first (step 2), then come back for the quiz.")
    else:
        rm_topic, rm_level = st.session_state.roadmap_for
        quiz = st.session_state.quiz
        if st.button("🧠 Generate quiz" if not quiz else "🔁 Make a new quiz", key="make_quiz"):
            if need_llm():
                data = run_ai(generate_quiz, rm_topic, rm_level, [m["title"] for m in rm["modules"]], QUIZ_SIZE)
                if data:
                    st.session_state.quiz = data
                    st.session_state.qid += 1
                    st.session_state.results = []
                    st.session_state.last_score = None
                    st.rerun()

        quiz = st.session_state.quiz
        if quiz:
            qid = st.session_state.qid
            st.write(f"Answer all {len(quiz)} questions. You need **{PASS_MARK}%** to pass.")
            with st.form(f"quiz_form_{qid}"):
                for i, q in enumerate(quiz):
                    st.markdown(f"**Q{i + 1}. {q['question']}**")
                    st.radio("Choose one", q["options"], index=None, key=f"q_{qid}_{i}",
                             label_visibility="collapsed")
                submitted = st.form_submit_button("✅ Submit answers")
            if submitted:
                answers = [st.session_state.get(f"q_{qid}_{i}") for i in range(len(quiz))]
                if None in answers:
                    st.warning("Please answer every question before submitting.")
                else:
                    results = []
                    for q, chosen in zip(quiz, answers):
                        correct = q["options"][q["answer_index"]]
                        results.append({"question": q["question"], "chosen": chosen, "correct": correct,
                                        "ok": chosen == correct, "explanation": q["explanation"]})
                    pct = round(100 * sum(r["ok"] for r in results) / len(results))
                    st.session_state.results = results
                    st.session_state.last_score = pct
                    st.session_state.best_score = max(st.session_state.best_score, pct)

            results = st.session_state.results
            if results:
                pct = st.session_state.last_score
                if pct >= PASS_MARK:
                    st.success(f"🎉 You scored {pct}%. Passed!")
                else:
                    st.warning(f"You scored {pct}%. You need {PASS_MARK}%. Review the answers below and try a new quiz.")
                for i, r in enumerate(results):
                    with st.expander(f"{'✅' if r['ok'] else '❌'} Q{i + 1}: {r['question']}"):
                        st.write(f"Your answer: {r['chosen']}")
                        if not r["ok"]:
                            st.write(f"Correct answer: **{r['correct']}**")
                        if r["explanation"]:
                            st.caption(r["explanation"])

# ================================================================ 4. BADGE
with tab_badge:
    if not rm:
        st.info("Make your roadmap first (step 2).")
    else:
        rm_topic, rm_level = st.session_state.roadmap_for
        rid = st.session_state.rid
        total = len(rm["modules"])
        done = sum(1 for i in range(total) if st.session_state.get(f"mod_{rid}_{i}", False))
        best = st.session_state.best_score
        m1, m2 = st.columns(2)
        m1.metric("Modules finished", f"{done}/{total}")
        m2.metric("Best quiz score", f"{best}%", help=f"You need {PASS_MARK}%")

        all_done = done == total
        passed = best >= PASS_MARK
        st.markdown(f"{'✅' if all_done else '⬜'} Finish every module in the roadmap")
        st.markdown(f"{'✅' if passed else '⬜'} Score at least {PASS_MARK}% in the quiz")

        if all_done and passed:
            if not user_name.strip():
                st.warning("Type your name in the sidebar to print it on your badge.")
            else:
                if not st.session_state.celebrated:
                    st.balloons()
                    st.session_state.celebrated = True
                png = make_badge_png(user_name.strip(), rm_topic, best, date.today().strftime("%d %b %Y"))
                st.image(png, width=380)
                st.download_button("⬇️ Download badge (PNG)", png,
                                   file_name=f"{rm_topic}-badge.png".replace(" ", "-"), mime="image/png")
                st.caption("This badge is made by this app for your own motivation. It is not an official certificate.")
        else:
            st.info("Finish both steps above to unlock your badge.")

# ================================================================ 5. ASSISTANT
with tab_chat:
    st.write("Ask anything about your topic. Example: *explain pointers with an analogy*.")
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask the study assistant...")
    if prompt:
        if need_llm():
            ctx_topic, ctx_level = st.session_state.roadmap_for or (topic, level)
            picked = st.session_state.picked
            course = picked["title"] if picked and picked["topic"] == ctx_topic else ""
            titles = [m["title"] for m in rm["modules"]] if rm else []
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                reply = run_ai(assistant_reply, ctx_topic, ctx_level, course, titles, st.session_state.chat)
                if reply:
                    st.markdown(reply)
                    st.session_state.chat.append({"role": "assistant", "content": reply})
                else:
                    st.session_state.chat.pop()  # keep user/assistant turns in pairs
