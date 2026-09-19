#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JEV 1.13 をセンター試験データセットで評価する。

データセット:
    center-finaltest/center-finaltest/   問題XML（正解ラベル <choice ra="yes"> は地理・地学のみ）
    torobo-data/extract/center-answer/   正答データ（11科目ぶんのマーク桁）

設計:
    1 大問（トップレベルの <question>）= 1 リクエスト。
        state     … 大問のリード文・資料（親から継承した instruction/data ＋ 大問自身のリード）
        questions … {"Q2": {"type": "choice",
                            "instructions": "問１ …。この問題の正解を，選択肢の中から一つ選べ。",
                            "criteria": {"1": "① …", ...}}, ...}

    JEV は画像を入力できないので、図は ［図 ファイル名］ というプレースホルダとして本文に残す。
    画像に依存する問題も既定で全部解かせ、あとから「図なし問題だけ」の統計を切り出せるようにする。

採点:
    レスポンスの probabilities（選択肢ごとの確率分布）が最大の選択肢を選んだことにして正誤を判定する。
    probabilities が無いときだけ choice を使う。
    正解した問題については「正解の確率 ÷ いちばん高い誤答の確率」を ratio として記録し、
    どれくらい余裕をもって正解したかを後から見られるようにする。

費用:
    OpenRouter のレスポンスに入っている usage.cost / usage.input_tokens / usage.output_tokens を
    そのまま集計する。大問（1リクエスト）単位で記録し、科目×年度×試験種別ごとに合計する。

得点:
    大問のリード文に書かれた「(配点 NN)」を満点として使う。
    小問ごとの配点は、正答データに <score> があればそれを使い、無ければ大問の配点を
    採点対象の小問数で等分する（近似。points_source に official / allocated と記録する）。

使い方:
    python jev_eval.py --dry-run                 # API を叩かず対象件数と概算コストだけ確認
    python jev_eval.py --simulate --limit 20     # API を叩かず統計処理の動作確認
    python jev_eval.py --subjects Chiri,Chigaku  # 科目を絞って本番実行
    python jev_eval.py --split all --workers 8   # 3分割（開発/開発テスト/最終テスト）を全部
    python jev_eval.py --official-scores official.json   # 実際の平均点・標準偏差を渡すと偏差値も出す

結果は JSONL（1 大問 1 行）とサマリ JSON に保存され、途中で止めても再実行で続きから走る。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
DEFAULT_INSTRUCTIONS = "この問題の正解を，選択肢の中から一つ選べ。"
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 524, 529}

# 分割ごとの (問題XML, 正答データ)。正答データが無い split では問題XMLの ra="yes" だけを使う。
SPLITS: dict[str, tuple[Path, Path | None]] = {
    "finaltest": (
        REPO_DIR / "center-finaltest" / "center-finaltest",
        REPO_DIR / "torobo-data" / "extract" / "center-answer" / "center-answer" / "finaltest",
    ),
    "devtest": (
        REPO_DIR / "torobo-data" / "extract" / "center-devtest" / "center-devtest",
        REPO_DIR / "torobo-data" / "extract" / "center-answer" / "center-answer" / "devtest",
    ),
    "dev": (
        REPO_DIR / "torobo-data" / "extract" / "center-dev" / "center-dev",
        REPO_DIR / "torobo-data" / "extract" / "center-answer" / "center-answer" / "dev",
    ),
}

# マークシートの桁 -> 選択肢番号。1〜9 が ①〜⑨、0 が ⑩。
MARK_DIGIT = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "0": 10}
POINTS_RE = re.compile(r"配点[^0-9]{0,6}([0-9]+)")
IMG_SOLVABLE_HINTS = ("見なくても", "見ずに")
# 全データ実行時の実測値（17,215問 / $0.3158）。--dry-run の概算に使う。
COST_PER_QUESTION_USD = 0.0000183


# --------------------------------------------------------------------------- #
# XML ユーティリティ
# --------------------------------------------------------------------------- #
def element_text(el: ET.Element) -> str:
    """要素を文字列化する。<br/> は改行、<img> は図のプレースホルダ、数式の重複表現
    （annotation-xml）はノイズなので落とす。"""

    def rec(e: ET.Element) -> str:
        if e.tag == "annotation-xml":
            return ""
        if e.tag == "br":
            return "\n"
        if e.tag == "img":
            src = e.get("src") or "?"
            comment = (e.get("comment") or "").strip()
            return f"［図 {src}］" if not comment else f"［図 {src}（{comment}）］"
        out = [e.text or ""]
        for child in e:
            out.append(rec(child))
            out.append(child.tail or "")
        return "".join(out)

    return rec(el)


def clean(text: str) -> str:
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n", text)
    return text.strip()


def image_srcs(el: ET.Element) -> list[str]:
    return [im.get("src") for im in el.iter("img") if im.get("src")]


def image_hints(el: ET.Element) -> list[str]:
    """「画像を見なくても解ける」等の注記（<img comment="...">）を集める。"""
    return [(im.get("comment") or "").strip() for im in el.iter("img") if im.get("comment")]


def lead_parts(el: ET.Element) -> tuple[str, list[str], list[str]]:
    """大問（または小問）直下のリード文・資料と、その画像 src / 注記を返す。
    入れ子になった <question> の中身は含めない。"""
    texts: list[str] = []
    srcs: list[str] = []
    hints: list[str] = []
    for child in el:
        if child.tag in ("instruction", "data"):
            texts.append(element_text(child))
            srcs += image_srcs(child)
            hints += image_hints(child)
            if child.tail:
                texts.append(child.tail)
        elif child.tag == "question":
            continue
        elif child.tail:
            texts.append(child.tail)
    return "\n".join(texts), srcs, hints


def declared_points(lead_text: str) -> float | None:
    m = POINTS_RE.search(lead_text)
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# 正答データ（center-answer）
# --------------------------------------------------------------------------- #
def parse_answer_file(path: Path) -> ET.Element:
    """正答データXMLを読む。1ファイルだけルート要素が欠けているので補修する。"""
    try:
        return ET.parse(path).getroot()
    except ET.ParseError:
        text = path.read_text(encoding="utf-8-sig")
        text = re.sub(r"^\s*<\?xml[^>]*\?>", "", text)
        text = re.sub(r"^\s*<!DOCTYPE[^>]*>", "", text).strip()
        text = re.sub(r"</answerTable>\s*$", "", text).strip()
        return ET.fromstring("<answerTable>" + text + "</answerTable>")


def load_answer_tables(answer_dir: Path | None) -> dict[tuple[str, str], list[dict]]:
    """{(ファイル名, 設問ID): [解答欄ごとの情報, ...]} を返す。"""
    tables: dict[tuple[str, str], list[dict]] = {}
    if not answer_dir or not answer_dir.exists():
        return tables
    for path in sorted(answer_dir.glob("*/*.xml")):
        root = parse_answer_file(path)
        stem = root.get("filename") or path.stem
        for data in root.findall("./data"):
            entry = {
                "column": (data.findtext("answer_column") or "").strip(),
                "answer": (data.findtext("answer") or "").strip(),
                "score": (data.findtext("score") or "").strip(),
                "style": (data.findtext("answer_style") or "").strip(),
                "qid": (data.findtext("question_ID") or "").strip(),
            }
            tables.setdefault((stem, entry["qid"]), []).append(entry)
    return tables


def answer_table_choice(entries: list[dict] | None, n_choices: int) -> tuple[str, float | None] | None:
    """正答データから「選択肢1つを選ぶ問題」の正解番号を復元する。
    復元できない（複数解答欄・複数正解・記号解答など）場合は None。"""
    if not entries or len(entries) != 1:
        return None
    entry = entries[0]
    if not entry["style"].startswith("multipleChoice"):
        return None
    if not re.fullmatch(r"[0-9]", entry["answer"]):
        return None
    idx = MARK_DIGIT[entry["answer"]]
    if idx > n_choices:  # 選択肢数と桁が食い違うものは捨てる
        return None
    try:
        score = float(entry["score"]) if entry["score"] else None
    except ValueError:
        score = None
    return str(idx), score


# --------------------------------------------------------------------------- #
# データセット -> 大問
# --------------------------------------------------------------------------- #
@dataclass
class Question:
    qid: str
    label: str
    instruction: str
    criteria: dict[str, str]
    answer: str
    answer_source: str  # "ra" | "answer_table"
    points: float
    points_source: str  # "official" | "allocated" | "default"
    n_choices: int
    has_image: bool
    image_solvable: bool
    missing_images: list[str] = field(default_factory=list)


@dataclass
class Section:
    key: str
    subject: str
    paper: str
    year: str
    exam: str
    file: str
    sid: str
    label: str
    state: str
    declared_points: float | None
    has_image: bool
    missing_images: list[str]
    questions: list[Question]


def make_criteria(choices: list[ET.Element]) -> dict[str, str]:
    criteria: dict[str, str] = {}
    for i, c in enumerate(choices, 1):
        text = clean(element_text(c))
        cnum = (c.findtext("cNum") or "").strip()
        criteria[str(i)] = text if text.startswith(cnum) else f"{cnum} {text}".strip()
    return criteria


def build_questions(
    top: ET.Element,
    subject_dir: Path,
    stem: str,
    answer_tables: dict[tuple[str, str], list[dict]],
    use_xml_answers: bool,
    use_table_answers: bool,
) -> list[Question]:
    """大問1つぶんの小問を集める。採点できる問題だけ返す。"""
    found: list[tuple[ET.Element, str, list[str], list[str]]] = []
    _, lead_srcs, lead_hints = lead_parts(top)
    if top.findall("./choices/choice"):
        # 大問そのものが設問の場合は、本文は state 側に入っているので instruction は空にする。
        found.append((top, "", lead_srcs, lead_hints))

    def handle(node: ET.Element, inherited_text: str, inherited_srcs: list[str], inherited_hints: list[str]) -> None:
        own_text, own_srcs, own_hints = lead_parts(node)
        ctx_text = f"{inherited_text}\n{own_text}"
        ctx_srcs = inherited_srcs + own_srcs
        ctx_hints = inherited_hints + own_hints
        found.append((node, ctx_text, ctx_srcs, ctx_hints))
        for child in node:
            if child.tag == "question":
                handle(child, ctx_text, ctx_srcs, ctx_hints)

    for child in top:
        if child.tag == "question":
            # 本文（大問のリード）は state 側に入るので、ここでは画像だけ引き継ぐ。
            handle(child, "", lead_srcs, lead_hints)

    questions: list[Question] = []
    seen: dict[str, int] = {}
    for node, ctx_text, ctx_srcs, ctx_hints in found:
        choices = node.findall("./choices/choice")
        if not choices:
            continue
        n_choices = len(choices)
        qid = node.get("id") or node.findtext("label") or "?"
        if qid in seen:  # 同一大問内で ID が重複したときの保険
            seen[qid] += 1
            qid = f"{qid}#{seen[qid]}"
        else:
            seen[qid] = 1

        answer: str | None = None
        source = ""
        official_points: float | None = None
        if use_xml_answers:
            ra = [i for i, c in enumerate(choices, 1) if c.get("ra") == "yes"]
            if len(ra) == 1:
                answer, source = str(ra[0]), "ra"
        if answer is None and use_table_answers:
            table = answer_table_choice(answer_tables.get((stem, node.get("id") or "")), n_choices)
            if table:
                answer, official_points, source = table[0], table[1], "answer_table"
        if answer is None:
            continue

        # ctx_srcs には自分の instruction/data の画像が入っているので、ここで足すのは選択肢ぶんだけ。
        srcs = ctx_srcs + [s for c in choices for s in image_srcs(c)]
        hints = ctx_hints + [h for c in choices for h in image_hints(c)]
        missing = [s for s in srcs if not (subject_dir / s).exists()]
        questions.append(
            Question(
                qid=qid,
                label=clean(node.findtext("label") or ""),
                instruction=clean(ctx_text),
                criteria=make_criteria(choices),
                answer=answer,
                answer_source=source,
                points=official_points or 0.0,
                points_source="official" if official_points else "unset",
                n_choices=n_choices,
                has_image=bool(srcs),
                image_solvable=any(h and any(hint in h for hint in IMG_SOLVABLE_HINTS) for h in hints),
                missing_images=sorted(set(missing)),
            )
        )
    return questions


def allocate_points(section: Section) -> None:
    """小問の配点を決める。正答データに <score> があればそれ、無ければ大問配点の等分。"""
    unknown = [q for q in section.questions if q.points_source != "official"]
    if section.declared_points and unknown:
        share = section.declared_points / len(unknown)
        for q in unknown:
            q.points = share
            q.points_source = "allocated"
    for q in section.questions:
        if q.points_source == "unset":
            q.points = 1.0
            q.points_source = "default"


def load_sections(
    split: str,
    subjects: set[str] | None,
    years: set[str] | None,
    use_xml_answers: bool,
    use_table_answers: bool,
) -> tuple[list[Section], dict[str, dict]]:
    """(採点対象の大問, ファイルごとの試験情報) を返す。

    試験情報には「その試験の満点（配点の合計）」と「問題数」を入れておく。
    採点できない小問は大問ごと落ちるので、満点はファイル側から別に数える必要がある。
    split="all" のときは開発用・開発テスト用・最終テスト用の3分割をまとめて読む。
    """
    targets = sorted(SPLITS) if split == "all" else [split]
    for name in targets:
        if not SPLITS[name][0].exists():
            raise SystemExit(f"データセットが見つかりません: {SPLITS[name][0]}")
    sections: list[Section] = []
    exam_meta: dict[str, dict] = {}
    for name in targets:
        part_sections, part_meta = _load_split(name, subjects, years, use_xml_answers, use_table_answers)
        sections += part_sections
        exam_meta.update(part_meta)
    sections.sort(key=lambda s: s.key)
    return sections, exam_meta


def _load_split(
    split: str,
    subjects: set[str] | None,
    years: set[str] | None,
    use_xml_answers: bool,
    use_table_answers: bool,
) -> tuple[list[Section], dict[str, dict]]:
    dataset_dir, answer_dir = SPLITS[split]
    tables = load_answer_tables(answer_dir) if use_table_answers else {}
    sections: list[Section] = []
    exam_meta: dict[str, dict] = {}
    for path in sorted(dataset_dir.glob("*/*.xml")):
        subject = path.parent.name
        if subjects and subject not in subjects:
            continue
        parts = path.stem.split("--")
        year = parts[0].split("-")[-1] if parts else ""
        exam = parts[1].split("-")[0] if len(parts) > 1 else ""
        paper = parts[1].split("-", 1)[1] if len(parts) > 1 and "-" in parts[1] else exam
        if years and year not in years:
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:  # 壊れたXMLはスキップ
            print(f"  [skip] {path.name}: {exc}", file=sys.stderr)
            continue
        tops = [c for c in root if c.tag == "question"]
        exam_meta[path.name] = {
            "subject": subject,
            "paper": paper,
            "year": year,
            "exam": exam,
            # 正解が分からない小問も含めた「試験全体の満点と問題数」
            "exam_points": sum(declared_points(lead_parts(t)[0]) or 0 for t in tops),
            "n_questions": len([q for q in root.iter("question") if q.findall("./choices/choice")]),
        }
        doc_text = "\n".join(element_text(c) for c in root if c.tag in ("instruction", "data"))
        doc_srcs = [s for c in root if c.tag in ("instruction", "data") for s in image_srcs(c)]
        for top in tops:
            lead, lead_srcs, _ = lead_parts(top)
            state = clean(f"{doc_text}\n{lead}")
            questions = build_questions(
                top, path.parent, path.stem, tables, use_xml_answers, use_table_answers
            )
            if not questions:
                continue
            sid = top.get("id") or top.findtext("label") or "?"
            section = Section(
                key=f"{path.name}:{sid}",
                subject=subject,
                paper=paper,
                year=year,
                exam=exam,
                file=path.name,
                sid=sid,
                label=clean(top.findtext("label") or ""),
                state=state,
                declared_points=declared_points(lead),
                has_image=bool(lead_srcs) or any(q.has_image for q in questions),
                missing_images=sorted({s for s in lead_srcs if not (path.parent / s).exists()}),
                questions=questions,
            )
            allocate_points(section)
            sections.append(section)
    return sections, exam_meta


# --------------------------------------------------------------------------- #
# OpenRouter Decisions API
# --------------------------------------------------------------------------- #
def load_api_key() -> str:
    env_path = REPO_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and ("APIKEY" in line.upper() or "API_KEY" in line.upper()):
                key = line.split("=", 1)[1].strip().strip("\"'")
                if key:
                    return key
    for name in ("OPENROUTER_APIKEY", "OPENROUTER_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    raise SystemExit(".env か環境変数に OPENROUTER_APIKEY を設定してください")


def question_payload(question: Question, instructions: str) -> dict:
    text = question.instruction
    prompt = f"{text}\n{instructions}" if text else instructions
    return {"type": "choice", "instructions": prompt, "criteria": question.criteria}


def request_decisions(api_key: str, payload: dict, timeout: float, retries: int) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        DECISIONS_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ushid/jev-center",
            "X-OpenRouter-Title": "jev-center-eval",
            "X-Title": "jev-center-eval",
        },
        method="POST",
    )
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return {"body": body, "elapsed": time.perf_counter() - started}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if exc.code in RETRY_STATUS and attempt < retries:
                time.sleep(2**attempt)
                continue
            return {"error": f"HTTP {exc.code}: {detail}", "elapsed": time.perf_counter() - started}
        except Exception as exc:  # noqa: BLE001 - 通信系はまとめて再試行
            if attempt < retries:
                time.sleep(2**attempt)
                continue
            return {"error": f"{type(exc).__name__}: {exc}", "elapsed": time.perf_counter() - started}
    return {"error": "unreachable"}


def ask_section(api_key: str, section: Section, model: str, instructions: str, timeout: float, retries: int) -> dict:
    """大問1つを1リクエストで解かせる。state に本文、questions に各小問。"""
    payload = {
        "model": model,
        "state": section.state,
        "questions": {q.qid: question_payload(q, instructions) for q in section.questions},
    }
    return request_decisions(api_key, payload, timeout, retries)


def ask_one(
    api_key: str, section: Section, question: Question, model: str, instructions: str, timeout: float, retries: int
) -> dict:
    """従来方式（1問1リクエスト）。比較用。"""
    payload = {
        "model": model,
        "state": clean(f"{section.state}\n{question.instruction}"),
        "questions": {"answer": question_payload(Question(**{**question.__dict__, "instruction": ""}), instructions)},
    }
    return request_decisions(api_key, payload, timeout, retries)


def normalize_response(result: dict) -> tuple[dict, str | None]:
    if result.get("error"):
        return {}, result["error"]
    answers = (result.get("body") or {}).get("answers") or {}
    return answers, None


def pick_prediction(answer: dict) -> tuple[str | None, dict | None]:
    """probabilities の確率が最大の選択肢を予測として選ぶ。無ければ choice を使う。"""
    probs = answer.get("probabilities")
    if isinstance(probs, dict) and probs:
        try:
            best = max(probs, key=lambda k: float(probs[k]))
            return best, probs
        except (TypeError, ValueError):
            pass
    return answer.get("choice"), None


def question_result(question: Question, answer: dict, error: str | None) -> dict:
    pred, probs = pick_prediction(answer)
    raw_choice = answer.get("choice")
    p_correct = p_best_wrong = ratio = None
    if probs:
        try:
            values = {k: float(v) for k, v in probs.items()}
            p_correct = values.get(question.answer)
            wrong = [v for k, v in values.items() if k != question.answer]
            p_best_wrong = max(wrong) if wrong else None
            if p_correct is not None and p_best_wrong:
                ratio = p_correct / p_best_wrong
        except (TypeError, ValueError):
            p_correct = p_best_wrong = ratio = None
    return {
        "qid": question.qid,
        "label": question.label,
        "answer": question.answer,
        "answer_source": question.answer_source,
        "pred": pred,
        "correct": (pred == question.answer) if error is None else None,
        "choice": raw_choice,  # API が返した choice（argmax と食い違う場合の確認用）
        "confidence": answer.get("confidence"),
        "probabilities": probs,
        "p_correct": p_correct,
        "p_best_wrong": p_best_wrong,
        "ratio": ratio,
        "points": question.points,
        "points_source": question.points_source,
        "n_choices": question.n_choices,
        "has_image": question.has_image,
        "image_solvable": question.image_solvable,
        "error": error,
    }


def section_row(
    section: Section,
    questions: list[dict],
    elapsed: float,
    cost: float,
    input_tokens: int,
    output_tokens: int,
    error: str | None,
    n_requests: int,
) -> dict:
    return {
        "key": section.key,
        "subject": section.subject,
        "paper": section.paper,
        "year": section.year,
        "exam": section.exam,
        "file": section.file,
        "sid": section.sid,
        "label": section.label,
        "state": section.state,
        "declared_points": section.declared_points,
        "has_image": section.has_image,
        "missing_images": section.missing_images,
        "elapsed": elapsed,
        "cost": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "n_requests": n_requests,
        "error": error,
        "questions": questions,
    }


def run_section(
    api_key: str,
    section: Section,
    model: str,
    instructions: str,
    timeout: float,
    retries: int,
    per_question: bool,
) -> dict:
    questions_out: list[dict] = []
    if per_question:
        elapsed = 0.0
        cost = in_tok = out_tok = 0.0
        errors: list[str] = []
        for question in section.questions:
            result = ask_one(api_key, section, question, model, instructions, timeout, retries)
            answers, error = normalize_response(result)
            usage = (result.get("body") or {}).get("usage", {}) or {}
            elapsed = max(elapsed, result.get("elapsed") or 0.0)
            cost += usage.get("cost") or 0.0
            in_tok += usage.get("input_tokens") or 0
            out_tok += usage.get("output_tokens") or 0
            if error:
                errors.append(f"{question.qid}: {error}")
            questions_out.append(question_result(question, answers.get("answer") or {}, error))
        return section_row(
            section, questions_out, elapsed, cost, in_tok, out_tok,
            "; ".join(errors) or None, len(section.questions),
        )

    result = ask_section(api_key, section, model, instructions, timeout, retries)
    answers, error = normalize_response(result)
    usage = (result.get("body") or {}).get("usage", {}) or {}
    for question in section.questions:
        got = answers.get(question.qid)
        questions_out.append(
            question_result(question, got or {}, error or ("response missing" if got is None else None))
        )
    return section_row(
        section, questions_out, result.get("elapsed") or 0.0,
        usage.get("cost") or 0.0, usage.get("input_tokens") or 0, usage.get("output_tokens") or 0,
        error, 1,
    )


# --------------------------------------------------------------------------- #
# 集計
# --------------------------------------------------------------------------- #
def block(rows: list[dict], section_rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    good = [r for r in ok if r.get("correct")]
    elapsed = [s["elapsed"] for s in section_rows if s.get("elapsed")]
    costs = [s.get("cost") or 0.0 for s in section_rows]
    score = sum(r.get("points") or 0.0 for r in good)
    max_score = sum(r.get("points") or 0.0 for r in ok)
    return {
        "n": len(ok),
        "errors": len(rows) - len(ok),
        "correct": len(good),
        "accuracy": (len(good) / len(ok)) if ok else 0.0,
        "score": score,
        "max_score": max_score,
        "score_rate": (score / max_score) if max_score else 0.0,
        "mean_latency_s": statistics.mean(elapsed) if elapsed else 0.0,
        "p50_latency_s": statistics.median(elapsed) if elapsed else 0.0,
        "max_latency_s": max(elapsed) if elapsed else 0.0,
        "total_cost_usd": sum(costs),
        "mean_cost_usd": (sum(costs) / len(section_rows)) if section_rows else 0.0,
        "input_tokens": sum(s.get("input_tokens") or 0 for s in section_rows),
        "output_tokens": sum(s.get("output_tokens") or 0 for s in section_rows),
        "n_requests": sum(s.get("n_requests") or 0 for s in section_rows),
    }


def flatten(rows: list[dict]) -> list[dict]:
    """大問行を小問ごとの行に展開する（大問の属性を継承させる）。"""
    q_rows: list[dict] = []
    for row in rows:
        sec_error = row.get("error")
        for q in row["questions"]:
            q_rows.append({
                **q,
                "subject": row["subject"],
                "paper": row.get("paper") or row["subject"],
                "year": row["year"],
                "exam": row["exam"],
                "key": f"{row['key']}:{q['qid']}",
                "error": q.get("error") or sec_error,
            })
    return q_rows


def official_lookup(official: dict | None, subject: str, year: str, exam: str) -> dict | None:
    if not official:
        return None
    node = official.get(subject, {}).get(year)
    if not node:
        return None
    if isinstance(node, dict) and exam in node:
        return node[exam]
    return node if isinstance(node, dict) else None


def top_margin(question: dict) -> float | None:
    """1位と2位の確率差。probabilities から毎回計算する。"""
    probs = question.get("probabilities")
    if not isinstance(probs, dict) or len(probs) < 2:
        return None
    try:
        values = sorted((float(v) for v in probs.values()), reverse=True)
    except (TypeError, ValueError):
        return None
    return values[0] - values[1]


def confidence_stats(qs: list[dict]) -> dict:
    """正解の確信度。ratio = 正解の確率 / いちばん高い誤答の確率。

    margin は「1位の確率 － 2位の確率」。予測は最上位の選択肢なので、
    正解確率そのものではなく margin で分けたときに実際の正解率がどう変わるかを見る。
    """
    ratios = [q["ratio"] for q in qs if q.get("ratio") is not None]
    margins = [m for m in (top_margin(q) for q in qs) if m is not None]
    pcs = [q["p_correct"] for q in qs if q.get("p_correct") is not None]
    correct = [q for q in qs if q.get("correct")]
    correct_ratios = [q["ratio"] for q in correct if q.get("ratio") is not None]
    zero_wrong = sum(1 for q in qs if q.get("p_best_wrong") == 0)

    def share(values: list[float], limit: float) -> float | None:
        return (sum(1 for v in values if v < limit) / len(values)) if values else None

    buckets = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0001)]
    calibration = []
    for lo, hi in buckets:
        group = [q for q in qs if (m := top_margin(q)) is not None and lo <= m < hi]
        calibration.append({
            "range": f"{lo:.2f}-{hi:.2f}",
            "n": len(group),
            "accuracy": (sum(1 for q in group if q["correct"]) / len(group)) if group else None,
        })
    return {
        "n": len(ratios),
        "mean_ratio": statistics.mean(ratios) if ratios else None,
        "median_ratio": statistics.median(ratios) if ratios else None,
        "mean_ratio_correct": statistics.mean(correct_ratios) if correct_ratios else None,
        "median_ratio_correct": statistics.median(correct_ratios) if correct_ratios else None,
        "share_ratio_lt_1_5": share(ratios, 1.5),
        "share_ratio_lt_3": share(ratios, 3.0),
        "mean_p_correct": statistics.mean(pcs) if pcs else None,
        "median_p_correct": statistics.median(pcs) if pcs else None,
        "mean_margin": statistics.mean(margins) if margins else None,
        "median_margin": statistics.median(margins) if margins else None,
        "n_zero_wrong": zero_wrong,  # 誤答の確率が全部 0（満票）だった件数
        "calibration": calibration,
    }


def summarize(rows: list[dict], official: dict | None = None, exam_meta: dict[str, dict] | None = None) -> dict:
    q_rows = flatten(rows)
    sec_rows = [r for r in rows if r["questions"]]

    def ok_qs(subset: list[dict]) -> list[dict]:
        return [q for q in subset if not q.get("error") and q.get("correct") is not None]

    def q_block(qs: list[dict], srows: list[dict]) -> dict:
        good = [q for q in qs if q["correct"]]
        score = sum(q["points"] for q in good)
        max_score = sum(q["points"] for q in qs)
        return {
            "n": len(qs), "correct": len(good),
            "accuracy": (len(good) / len(qs)) if qs else 0.0,
            "score": score, "max_score": max_score,
            "score_rate": (score / max_score) if max_score else 0.0,
            "n_sections": len(srows),
        }

    subjects = sorted({r["subject"] for r in rows})
    exams = sorted({(r["subject"], r.get("paper") or r["subject"], r["year"], r["exam"]) for r in rows})

    by_year = []
    for subject, paper, year, exam in exams:
        rs = [
            r for r in rows
            if (r["subject"], r.get("paper") or r["subject"], r["year"], r["exam"]) == (subject, paper, year, exam)
        ]
        qs = ok_qs([q for r in rs for q in r["questions"]])
        txt = [q for q in qs if not q["has_image"]]
        img = [q for q in qs if q["has_image"]]
        files = sorted({r["file"] for r in rs})
        meta = [exam_meta[f] for f in files if exam_meta and f in exam_meta]
        entry = {
            "subject": subject, "paper": paper, "year": year, "exam": exam,
            "n_files": len(files),
            "n_sections": len(rs), "n": len(qs),
            "correct": sum(1 for q in qs if q["correct"]),
            "accuracy": (sum(1 for q in qs if q["correct"]) / len(qs)) if qs else 0.0,
            "score": sum(q["points"] for q in qs if q["correct"]),
            "max_score": sum(q["points"] for q in qs),
            # 試験全体（採点できない小問も含む）の満点と問題数
            "exam_max_score": sum(m["exam_points"] for m in meta),
            "exam_questions": sum(m["n_questions"] for m in meta),
            "text_only_n": len(txt),
            "text_only_accuracy": (sum(1 for q in txt if q["correct"]) / len(txt)) if txt else None,
            "image_n": len(img),
            "image_accuracy": (sum(1 for q in img if q["correct"]) / len(img)) if img else None,
        }
        entry["score_rate"] = entry["score"] / entry["max_score"] if entry["max_score"] else 0.0
        entry["coverage"] = entry["max_score"] / entry["exam_max_score"] if entry["exam_max_score"] else None
        entry["n_ungraded"] = max(entry["exam_questions"] - entry["n"], 0)
        # 1 年度 1 科目（1 試験）を解くのにかかった費用と時間
        entry["total_cost_usd"] = sum(r.get("cost") or 0.0 for r in rs)
        entry["mean_cost_usd"] = entry["total_cost_usd"] / len(qs) if qs else None
        entry["n_requests"] = sum(r.get("n_requests") or 0 for r in rs)
        entry["total_elapsed_s"] = sum(r.get("elapsed") or 0.0 for r in rs)
        entry["input_tokens"] = sum(r.get("input_tokens") or 0 for r in rs)
        entry["output_tokens"] = sum(r.get("output_tokens") or 0 for r in rs)
        official_row = official_lookup(official, subject, year, exam)
        if official_row and official_row.get("std"):
            entry["official_mean"] = official_row.get("mean")
            entry["official_std"] = official_row.get("std")
            if official_row.get("mean") is not None:
                # 満点が違うので、得点率を実際の満点に換算してから偏差値を出す
                official_total = official_row.get("total") or entry["exam_max_score"] or entry["max_score"]
                entry["scaled_score"] = entry["score_rate"] * official_total
                entry["hensachi"] = 50 + 10 * (entry["scaled_score"] - official_row["mean"]) / official_row["std"]
        by_year.append(entry)

    return {
        "model": MODEL,
        "overall": block(q_rows, sec_rows),
        "by_subject": {
            s: {
                **block([q for q in q_rows if q["subject"] == s], [r for r in sec_rows if r["subject"] == s]),
                # 科目ごとの 図なし / 図あり の内訳（README の表はこれを使う）
                "text_only": q_block(
                    ok_qs([q for q in q_rows if q["subject"] == s and not q["has_image"]]),
                    [r for r in sec_rows if r["subject"] == s and not r["has_image"]],
                ),
                "image_required": q_block(
                    ok_qs([q for q in q_rows if q["subject"] == s and q["has_image"]]),
                    [r for r in sec_rows if r["subject"] == s and r["has_image"]],
                ),
            }
            for s in subjects
        },
        "by_exam_type": {
            e: block([q for q in q_rows if q["exam"] == e], [r for r in sec_rows if r["exam"] == e])
            for e in sorted({r["exam"] for r in rows})
        },
        "by_image": {
            "text_only": q_block(ok_qs([q for q in q_rows if not q["has_image"]]), [r for r in sec_rows if not r["has_image"]]),
            "image_required": q_block(ok_qs([q for q in q_rows if q["has_image"]]), [r for r in sec_rows if r["has_image"]]),
            "image_annotated_solvable": q_block(
                ok_qs([q for q in q_rows if q["image_solvable"]]),
                [r for r in sec_rows if any(q["image_solvable"] for q in r["questions"])],
            ),
        },
        "confidence": {
            "all": confidence_stats(ok_qs(q_rows)),
            "correct_only": confidence_stats([q for q in ok_qs(q_rows) if q["correct"]]),
            "text_only": confidence_stats(ok_qs([q for q in q_rows if not q["has_image"]])),
            "image_required": confidence_stats(ok_qs([q for q in q_rows if q["has_image"]])),
            # probabilities の argmax と API の choice が食い違った件数
            "argmax_mismatch": sum(
                1 for q in ok_qs(q_rows) if q.get("choice") and q.get("pred") and q["choice"] != q["pred"]
            ),
        },
        "by_subject_year_exam": by_year,
    }


def print_summary(summary: dict) -> None:
    print("\n== 全体 ==")
    for title, stats in [("全体", summary["overall"])] + list(summary["by_subject"].items()):
        if not stats["n"]:
            continue
        print(
            f"  {title:<13}{stats['accuracy'] * 100:6.1f}%  ({stats['correct']}/{stats['n']}, 失敗 {stats['errors']})  "
            f"得点 {stats['score']:.1f}/{stats['max_score']:.0f} ({stats['score_rate'] * 100:.1f}%)  "
            f"平均 {stats['mean_latency_s']:.2f}s / p50 {stats['p50_latency_s']:.2f}s  "
            f"${stats['total_cost_usd']:.4f}  in {stats['input_tokens']} out {stats['output_tokens']}"
        )

    print("\n== 図の有無で比較 ==")
    for title, stats in summary["by_image"].items():
        if not stats["n"]:
            continue
        print(
            f"  {title:<26}{stats['accuracy'] * 100:6.1f}%  ({stats['correct']}/{stats['n']})  "
            f"得点率 {stats['score_rate'] * 100:5.1f}%  大問 {stats['n_sections']}"
        )

    print("\n== 試験種別（本試験/追試験） ==")
    for title, stats in summary["by_exam_type"].items():
        if not stats["n"]:
            continue
        print(f"  {title:<10}{stats['accuracy'] * 100:6.1f}%  ({stats['correct']}/{stats['n']})")

    print("\n== 年度別得点 ==")
    print(
        f"{'科目':<12}{'年度':>5} {'種別':<6}{'大問':>5}{'採点/全体':>10}{'正答':>6}"
        f"{'得点/満点':>15}{'得点率':>8}{'正解率':>8}{'図なし':>8}{'図あり':>8}{'偏差値':>9}"
    )
    for e in summary["by_subject_year_exam"]:
        txt = e["text_only_accuracy"]
        img = e["image_accuracy"]
        hen = e.get("hensachi")
        print(
            f"{e['paper']:<12}{e['year']:>5} {e['exam']:<6}{e['n_sections']:>5}"
            f"{str(e['n']) + '/' + str(e['exam_questions']):>10}{e['correct']:>6}"
            f"{e['score']:>7.1f}/{e['max_score']:<7.0f}{e['score_rate'] * 100:>7.1f}%{e['accuracy'] * 100:>7.1f}%"
            + (f"{txt * 100:>7.1f}%" if txt is not None else f"{'-':>8}")
            + (f"{img * 100:>7.1f}%" if img is not None else f"{'-':>8}")
            + (f"{hen:>9.1f}" if hen is not None else "")
        )
    print(
        "  ※ 得点/満点 は採点できた小問の合計（配点の合計）。"
        "「採点/全体」は採点対象になった小問数 / その試験の問題数。"
    )

    print("\n== 1年度1科目ごとの費用と時間 ==")
    print(f"{'科目':<12}{'年度':>5} {'種別':<6}{'問題':>6}{'リクエスト':>10}{'費用$':>11}{'1問$':>10}"
          f"{'合計秒':>9}{'1リクエスト秒':>13}{'in tok':>10}{'out tok':>9}")
    for e in summary["by_subject_year_exam"]:
        per_req = (e["total_elapsed_s"] / e["n_requests"]) if e["n_requests"] else 0.0
        per_q = e["mean_cost_usd"] or 0.0
        print(
            f"{e['paper']:<12}{e['year']:>5} {e['exam']:<6}{e['n']:>6}{e['n_requests']:>10}"
            f"{e['total_cost_usd']:>11.4f}{per_q:>10.6f}{e['total_elapsed_s']:>9.1f}{per_req:>13.2f}"
            f"{e['input_tokens']:>10}{e['output_tokens']:>9}"
        )
    total_cost = sum(e["total_cost_usd"] for e in summary["by_subject_year_exam"])
    total_q = sum(e["n"] for e in summary["by_subject_year_exam"])
    total_req = sum(e["n_requests"] for e in summary["by_subject_year_exam"])
    print(
        f"  合計 {total_q} 問 / {total_req} リクエスト / ${total_cost:.4f}"
        + (f"（1問 ${total_cost / total_q:.6f}）" if total_q else "")
    )

    conf = summary.get("confidence") or {}
    if conf.get("all", {}).get("n"):
        c_all, c_ok = conf["all"], conf.get("correct_only", {})
        print("\n== 確信度（正解の確率 ÷ いちばん高い誤答の確率） ==")
        print(
            f"  全体     中央値 {c_all['median_ratio']:.2f}（1.5倍未満 {(c_all['share_ratio_lt_1_5'] or 0) * 100:.1f}%"
            f" / 3倍未満 {(c_all['share_ratio_lt_3'] or 0) * 100:.1f}%）"
            f" 正解確率と誤答の差の中央値 {c_all['median_margin']:.3f}"
        )
        if c_ok.get("median_ratio") is not None:
            print(f"  正解のみ 中央値 {c_ok['median_ratio']:.2f} / 正解確率と誤答の差の中央値 {c_ok['median_margin']:.3f}")
        print(f"  正解確率の中央値 {c_all['median_p_correct']:.3f}")
        for title in ("text_only", "image_required"):
            c = conf.get(title) or {}
            if c.get("median_ratio") is not None:
                print(f"  {title:<24}中央値 {c['median_ratio']:.2f} / 正解確率の中央値 {c['median_p_correct']:.3f}")
        print("  1位と2位の確率差ごとの実際の正解率:")
        for row in c_all["calibration"]:
            if row["n"]:
                print(f"    確率差 {row['range']}  n={row['n']:<6} 実際の正解率 {row['accuracy'] * 100:5.1f}%")
        if conf.get("argmax_mismatch"):
            print(f"  ※ probabilities の argmax と API の choice が食い違った件数: {conf['argmax_mismatch']}")


# --------------------------------------------------------------------------- #
# シミュレーション（API を使わない動作確認）
# --------------------------------------------------------------------------- #
def simulate_rows(sections: list[Section], accuracy: float = 0.88) -> list[dict]:
    """API を使わずに統計処理を検証するための擬似結果。probabilities も作る。"""
    rows = []
    for section in sections:
        rng = random.Random(section.key)
        questions = []
        for q in section.questions:
            keys = [str(i) for i in range(1, q.n_choices + 1)]
            hit = rng.random() < accuracy
            p_correct = min(0.999, max(0.05, rng.gauss(0.75, 0.25))) if hit else rng.uniform(0.01, 0.45)
            rest = (1.0 - p_correct) / max(len(keys) - 1, 1)
            probs = {k: round(rest, 6) for k in keys}
            probs[q.answer] = round(p_correct, 6)
            pred = max(probs, key=lambda k: probs[k])
            wrong = [v for k, v in probs.items() if k != q.answer]
            p_best_wrong = max(wrong) if wrong else None
            questions.append({
                "qid": q.qid, "label": q.label, "answer": q.answer, "answer_source": q.answer_source,
                "pred": pred, "correct": pred == q.answer, "choice": pred,
                "confidence": p_correct, "probabilities": probs,
                "p_correct": probs[q.answer], "p_best_wrong": p_best_wrong,
                "ratio": (probs[q.answer] / p_best_wrong) if p_best_wrong else None,
                "points": q.points, "points_source": q.points_source,
                "n_choices": q.n_choices, "has_image": q.has_image,
                "image_solvable": q.image_solvable, "error": None,
            })
        rows.append({
            "key": section.key, "subject": section.subject, "paper": section.paper,
            "year": section.year, "exam": section.exam,
            "file": section.file, "sid": section.sid, "label": section.label, "state": section.state,
            "declared_points": section.declared_points, "has_image": section.has_image,
            "missing_images": section.missing_images, "elapsed": 0.8,
            "cost": 0.00004 * len(questions), "input_tokens": 800 * len(questions),
            "output_tokens": 30 * len(questions), "n_requests": 1, "error": None,
            "simulated": True, "questions": questions,
        })
    return rows


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="JEV 1.13 をセンター試験データセットで評価する")
    parser.add_argument("--split", default="all", choices=["all", *sorted(SPLITS)],
                        help="使う分割。all=開発/開発テスト/最終テストの3分割すべて")
    parser.add_argument("--model", default=MODEL, help=f"モデル名（既定: {MODEL}）")
    parser.add_argument("--subjects", help="科目をカンマ区切りで指定（例: Chiri,Chigaku）")
    parser.add_argument("--years", help="年度をカンマ区切りで指定（例: 1990,1992）")
    parser.add_argument("--limit", type=int, help="評価する大問数（ランダム抽出）")
    parser.add_argument("--seed", type=int, default=0, help="--limit の抽出シード")
    parser.add_argument("--workers", type=int, default=4, help="並列リクエスト数")
    parser.add_argument("--text-only", action="store_true", help="図を参照する問題を除く")
    parser.add_argument("--per-question", action="store_true", help="大問バッチではなく1問1リクエストで送る")
    parser.add_argument("--answers", default="auto", choices=["auto", "xml", "table"],
                        help="正解の取得元。auto=問題XMLの ra と正答データの両方")
    parser.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS, help="choice 質問の instructions")
    parser.add_argument("--out", default=str(REPO_DIR / "results" / "jev_sections.jsonl"), help="結果の出力先 (JSONL)")
    parser.add_argument("--timeout", type=float, default=120.0, help="1 リクエストのタイムアウト秒")
    parser.add_argument("--retries", type=int, default=3, help="失敗時の再試行回数")
    parser.add_argument("--dry-run", action="store_true", help="API を叩かず対象件数と概算だけ出す")
    parser.add_argument("--simulate", action="store_true", help="API を叩かず擬似解答で統計処理を検証する")
    parser.add_argument("--official-scores", help="実際の平均点・標準偏差の JSON（偏差値の算出用）")
    parser.add_argument("--redo", action="store_true", help="既存の結果を無視して最初から実行する")
    args = parser.parse_args()

    subjects = set(args.subjects.split(",")) if args.subjects else None
    years = set(args.years.split(",")) if args.years else None
    use_xml = args.answers in ("auto", "xml")
    use_table = args.answers in ("auto", "table")

    sections, exam_meta = load_sections(args.split, subjects, years, use_xml, use_table)
    if args.text_only:
        # 図を参照する「小問」だけを外す。大問のリード文に図があっても、
        # 図を見ずに解ける小問は残したいため、大問単位では切らない。
        for s in sections:
            s.questions = [q for q in s.questions if not q.has_image]
        sections = [s for s in sections if s.questions]
    if args.limit and args.limit < len(sections):
        sections = random.Random(args.seed).sample(sections, args.limit)
        sections.sort(key=lambda s: s.key)
    n_questions = sum(len(s.questions) for s in sections)
    n_points = sum(s.declared_points or sum(q.points for q in s.questions) for s in sections)
    exam_points = sum(m["exam_points"] for m in exam_meta.values())
    exam_questions = sum(m["n_questions"] for m in exam_meta.values())
    sources: dict[str, int] = {}
    for s in sections:
        for q in s.questions:
            sources[q.answer_source] = sources.get(q.answer_source, 0) + 1
    print(
        f"対象: {len(sections)} 大問 / {n_questions} 問（全 {exam_questions} 問中）"
        f" / 採点対象の配点合計 {n_points:.0f} 点（全 {exam_points:.0f} 点中）"
        f"（正解の出所: {sources}）"
    )
    if args.dry_run:
        est_cost = n_questions * COST_PER_QUESTION_USD
        print(f"  概算コスト: 実測（1問 ${COST_PER_QUESTION_USD}）から ${est_cost:.4f}")
        per_sec = args.workers / 0.51  # 実測の 1 リクエスト平均 0.51 秒
        print(f"  概算時間: 1大問 0.51s として {len(sections) / per_sec / 60:.1f} 分（workers={args.workers}）")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.redo and out_path.exists():
        out_path.unlink()

    done: dict[str, dict] = {}
    if out_path.exists() and not args.redo:
        legacy = 0
        for line in out_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "key" not in row or "questions" not in row:
                legacy += 1  # 旧版（1問1行）の結果が混ざっている場合は無視する
                continue
            done[row["key"]] = row
        if legacy:
            print(
                f"  [warn] 旧形式の行 {legacy} 件を無視しました。"
                f" 混ざったまま再開すると集計がずれるので、本番は新しい --out を使うのがおすすめです"
            )
        print(f"既存の結果 {len(done)} 大問を読み込みました（再開モード）")

    started = time.perf_counter()
    requests_this_run = 0
    if args.simulate:
        for row in simulate_rows(sections):
            done[row["key"]] = row
        print("※ --simulate: API は叩かず、擬似解答で統計処理だけ検証しています")
    else:
        todo = [s for s in sections if s.key not in done]
        api_key = load_api_key()
        if todo:
            with out_path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
                futures = {
                    pool.submit(
                        run_section, api_key, s, args.model, args.instructions,
                        args.timeout, args.retries, args.per_question
                    ): s
                    for s in todo
                }
                for i, future in enumerate(as_completed(futures), 1):
                    row = future.result()
                    requests_this_run += row.get("n_requests") or 0
                    done[row["key"]] = row
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    n_ok = sum(1 for q in row["questions"] if q["correct"])
                    mark = "!" if row["error"] else "o"
                    print(
                        f"[{i}/{len(todo)}] {mark} {row['subject']} {row['year']} {row['sid']} "
                        f"{n_ok}/{len(row['questions'])} 問正解 {row['elapsed']:.2f}s ${row['cost'] or 0:.6f}",
                        flush=True,
                    )

    rows = [done[s.key] for s in sections if s.key in done]
    official = (
        json.loads(Path(args.official_scores).read_text(encoding="utf-8-sig"))
        if args.official_scores
        else None
    )
    summary = summarize(rows, official, exam_meta)
    elapsed_this_run = time.perf_counter() - started
    summary["n_requests_this_run"] = requests_this_run
    # リクエストを 1 つも出さなかった再集計では、前回の実行時間をそのまま残す
    summary_path = out_path.with_suffix(".summary.json")
    previous_wall = None
    if summary_path.exists():
        try:
            previous_wall = json.loads(summary_path.read_text(encoding="utf-8-sig")).get("wall_clock_s")
        except (json.JSONDecodeError, OSError):
            previous_wall = None
    summary["wall_clock_s"] = elapsed_this_run if (requests_this_run or previous_wall is None) else previous_wall
    # リクエストの所要時間の合計（直列換算）。並列度に関係なく比較できる
    summary["total_request_seconds"] = sum(r.get("elapsed") or 0.0 for r in rows)
    summary["settings"] = {
        "split": args.split,
        "subjects": sorted(subjects) if subjects else "all",
        "text_only": args.text_only,
        "per_question": args.per_question,
        "answers": args.answers,
        "simulated": args.simulate,
    }
    print_summary(summary)
    if args.simulate:
        print("\n※ シミュレーションなので結果ファイルには保存していません")
    else:
        summary_path = out_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n結果: {out_path}\nサマリ: {summary_path}")


if __name__ == "__main__":
    main()
