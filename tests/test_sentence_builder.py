import pytest
from core.sentence_builder import SentenceBuilder

def test_initial_state():
    sb = SentenceBuilder()
    assert sb.get_display_text() == ""
    assert sb.current_word == ""
    assert sb.sentence == ""

def test_update_holds_sign_for_confirmation():
    sb = SentenceBuilder()
    sb.confirm_duration = 1.5
    
    # Not held long enough
    sb.update("A", 1.0)
    assert sb.get_display_text() == ""
    sb.update("A", 2.0)
    assert sb.get_display_text() == ""
    
    # Held long enough
    sb.update("A", 2.5)
    assert sb.get_display_text() == "A"

def test_update_resets_on_different_sign():
    sb = SentenceBuilder()
    sb.confirm_duration = 1.5
    sb.update("A", 1.0)
    sb.update("B", 2.0)  # changes sign
    sb.update("B", 3.5)  # confirms B
    assert sb.get_display_text() == "B"

def test_add_space():
    sb = SentenceBuilder()
    sb.confirm_duration = 1.5
    sb.update("H", 1.0)
    sb.update("H", 2.5)
    sb.update("I", 3.0)
    sb.update("I", 4.5)
    
    assert sb.get_display_text() == "HI"
    sb.add_space()
    assert sb.sentence == "HI "
    assert sb.current_word == ""
    assert sb.get_display_text() == "HI "

def test_clear():
    sb = SentenceBuilder()
    sb.confirm_duration = 1.5
    sb.update("A", 1.0)
    sb.update("A", 2.5)
    sb.clear()
    assert sb.get_display_text() == ""
    assert sb.current_word == ""
    assert sb.sentence == ""
