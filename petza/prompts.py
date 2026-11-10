"""Marking prompts: the instructions the online AI follows when it picks the outlines.

Several named prompts live in data/prompts.json; one is active. The local model
does not read text (it learns from the marked examples); an optional YAML block
at the top of a prompt still sets the measurable rules (see rules.py).
"""
from __future__ import annotations

import json
import os
import threading

from . import config
from .rules import Rules, parse_rules

DEFAULT_NAME = "ברירת מחדל"
OLD_FIRST_LINE = "אתה בודק סימון של פצע בתמונה רפואית. הקו הירוק הוא הסימון האוטומטי."
NEW_FIRST_LINE = "אתה מחליט על סימון של פצע בתמונה רפואית: לכל אזור, האם הוא פצע, ואיזה קו (A, B או C) עובר בדיוק על גבול הפצע."

DEFAULT_TEXT = """אתה מחליט על סימון של פצע בתמונה רפואית: לכל אזור, האם הוא פצע, ואיזה קו (A, B או C) עובר בדיוק על גבול הפצע.

## מה נחשב פצע (בתוך הקו)
1. רקמת גרנולציה: אדומה עד אדומה-כהה, לחה ומבריקה, עם מרקם גרעיני כמו אבני מרצפת קטנות.
2. סלאף (רקמה מתה רכה): צהבהבה, לבנבנה או אפרפרה, רכה, סיבית או דביקה, לרוב צמודה לבסיס הפצע.
3. נמק (אסכר): שחור, חום כהה או אפור, יבש, קשה ודומה לעור עבה.
4. כל אזור שבו השכבה העליונה של העור חסרה והרקמה שמתחת חשופה.

## מה לא נחשב פצע (מחוץ לקו)
1. עור יבש: לבנבן-אפור, מט, מתקלף בקשקשים דקים. פני העור שלמים, ורואים דרכם קווי עור ונקבוביות.
2. אזור שכבר החלים (עור חדש): ורוד בהיר עד צבע עור, חלק, יבש, מבריק מעט, בלי מרקם גרעיני. העור סגור.
3. עור אדום (אודם): אדום או סגלגל, אבל פני העור שלמים ויבשים, עם קווי עור ונקבוביות. אין רקמה חשופה.
4. השוליים הוורודים החיצוניים: פס ורוד בהיר, דק וחלק סביב הפצע, שבו עור חדש מתקדם פנימה. הוא לא חלק מהפצע.
5. עור לבן ונפוח מרטיבות (מצרציה) סביב השוליים: לבן, רך ומקומט.
6. כל דבר אחר: תחבושת, פלסטר, סרגל, מדבקה, כפפה, צל או השתקפות אור.

## איך מבחינים
- רצף פני העור: אם רואים קווי עור, נקבוביות או שיערות, העור שלם. זה לא פצע, גם אם הוא אדום.
- לחות וברק: רקמה פתוחה לחה ומבריקה. עור יבש, אזור שהחלים ועור אדום הם מטים או מבריקים מעט ואחידים.
- מרקם: גרנולציה גרעינית, סלאף סיבי, נמק קשה ויבש. עור שלם חלק או מקומט בעדינות.
- עומק: לפצע יש לרוב שפה ברורה ומדרגה קלה כלפי פנים. השוליים הוורודים שטוחים.
- צבע לבד לא מספיק: אדום יכול להיות גרנולציה (לחה וגרעינית) או אודם (יבש ושלם). מחליטים לפי מרקם ולחות.

## איך הקו צריך לעבור
- בדיוק על הגבול בין הרקמה הפתוחה לבין העור השלם או האזור שהחלים.
- לא חותך לתוך הפצע ולא משאיר חלק ממנו בחוץ.
- לא כולל עור בריא, שוליים ורודים, עור יבש, אודם או תחבושת.
- כל פצע נפרד מסומן בקו נפרד.
"""

FILE = config.DATA / "prompts.json"
_lock = threading.Lock()


def load() -> dict:
    store = {"active": DEFAULT_NAME, "prompts": {}}
    if FILE.is_file():
        try:
            data = json.loads(FILE.read_text(encoding="utf-8"))
            store["active"] = data.get("active") or DEFAULT_NAME
            store["prompts"] = {str(k): str(v) for k, v in (data.get("prompts") or {}).items()}
        except ValueError:
            pass
    store["prompts"].setdefault(DEFAULT_NAME, DEFAULT_TEXT)
    old = store["prompts"][DEFAULT_NAME]
    if old.startswith(OLD_FIRST_LINE):  # the default from before the AI picked outlines
        store["prompts"][DEFAULT_NAME] = NEW_FIRST_LINE + old[len(OLD_FIRST_LINE):]
    if store["active"] not in store["prompts"]:
        store["active"] = DEFAULT_NAME
    return store


def _save(store: dict) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, FILE)


def put(name: str, text: str) -> dict:
    """Create or update a prompt and make it the active one."""
    with _lock:
        store = load()
        store["prompts"][name] = text
        store["active"] = name
        _save(store)
        return store


def select(name: str) -> dict:
    with _lock:
        store = load()
        if name not in store["prompts"]:
            raise KeyError(name)
        store["active"] = name
        _save(store)
        return store


def delete(name: str) -> dict:
    with _lock:
        store = load()
        if name == DEFAULT_NAME:
            raise ValueError("The default prompt can be edited or restored, not deleted.")
        store["prompts"].pop(name, None)
        if store["active"] == name:
            store["active"] = DEFAULT_NAME
        _save(store)
        return store


def active_rules() -> Rules:
    store = load()
    return parse_rules(store["prompts"][store["active"]])
