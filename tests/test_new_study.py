# study page: defaults finish to a valid case, a bad answer blocks Next, cards write answers
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no window system needed; must precede the Qt import
from PySide6.QtWidgets import QApplication
from src.run.case_file import part_free_case
from src.run.study import study_choices
from app.new_study import (StudyPage, wizard_choices, wizard_labels, card_titles, card_descriptions, card_fields,
                           problem_optional_fields, flow_required_fields, flow_optional_fields)

application = QApplication.instance() or QApplication([])

class FakePreview:
    """
    A preview double with no VTK window, recording the last case file it was asked to draw.
    """

    def __init__(self):

        self.last_case_file = None

    def draw(self, case_file, reset_camera=False):

        self.last_case_file = case_file

        return []

    def close(self):

        pass

def make_page():
    """
    A StudyPage with a fake preview, so the tests need no render window.

    Returns the StudyPage.
    """

    return StudyPage({}, lambda: ".", lambda: [], lambda: (600.0, False, None), preview=FakePreview())

# test 1: the defaults give a valid, part-free case and the verdict shows the capped Re
def test_defaults_finish():

    page = make_page()
    case_file = page.case_file()
    part_free_case(case_file).validate()

    assert page.error is None
    assert "capped" in page.verdict.text()
    assert "memory =" in page.verdict.text()
    assert case_file.geometry == []

# test 2: an answer the engine rejects shows the error and blocks Next
def test_bad_answer_blocks():

    page = make_page()
    page.answers.problem = "rocket"
    page.refresh()

    assert page.error is not None
    assert "rocket" in page.verdict.text()
    assert not page.next_button.isEnabled()

# test 3: Next walks the steps; the last step reads Create and emits the finished case
def test_next_walks_to_create():

    page = make_page()
    created = []
    page.created.connect(lambda case_file: created.append(case_file))
    page.next_button.click()
    page.next_button.click()
    page.next_button.click()

    assert page.next_button.text() == "Create"
    page.next_button.click()

    assert len(created) == 1
    assert created[0].geometry == []

# test 4: a card click writes the answer and redraws the preview with the new case
def test_card_click_writes_answer():

    page = make_page()
    page.choose("problem", "ground_moving")

    assert page.answers.problem == "ground_moving"
    assert page.preview.last_case_file.domain.floor == "moving"

# test 5: Open case emits an empty path, and activating a recent item emits its path
def test_open_and_recent_signals():

    page = make_page()
    opened = []
    page.open_requested.connect(lambda path: opened.append(path))
    page.open_button.click()
    page.set_recent(["C:/x/a.json"])
    page.recent_list.itemActivated.emit(page.recent_list.item(0))

    assert opened == ["", "C:/x/a.json"]

# test 6: every shown field and choice has a label, and every card value has a title and a description
def test_every_field_and_card_labelled():

    form_fields = (*problem_optional_fields, *flow_required_fields, *flow_optional_fields)
    shown_fields = set(card_fields) | set(form_fields)
    shown_choices = {value for name in shown_fields if name in wizard_choices for value in wizard_choices[name]}
    missing = (shown_fields | shown_choices) - set(wizard_labels)

    assert missing == set()
    for name in card_fields:
        for value in study_choices[name]:
            assert value in card_titles
            assert value in card_descriptions
