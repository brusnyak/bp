import pytest
from playwright.sync_api import Page, expect

BASE_URL = "https://localhost:8000/ui/live-speech/live.html"

@pytest.mark.ui
def test_theme_toggle_functionality(page: Page):
    """
    Test to verify the theme toggle switches between light and dark themes.
    """
    page.goto(BASE_URL)

    theme_toggle_button = page.locator("#theme-toggle")
    expect(theme_toggle_button).to_be_visible()

    # Initial theme is light (default)
    expect(page.locator("html")).to_have_attribute("data-theme", "light")
    expect(theme_toggle_button.locator(".material-symbols-outlined")).to_have_text("dark_mode")
    print("UI Test: Initial theme is light.")

    # Toggle to dark theme
    theme_toggle_button.click()
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    expect(theme_toggle_button.locator(".material-symbols-outlined")).to_have_text("light_mode")
    print("UI Test: Toggled to dark theme.")

    # Toggle back to light theme
    theme_toggle_button.click()
    expect(page.locator("html")).to_have_attribute("data-theme", "light")
    expect(theme_toggle_button.locator(".material-symbols-outlined")).to_have_text("dark_mode")
    print("UI Test: Toggled back to light theme.")

@pytest.mark.ui
def test_pipeline_initialization_and_start_stop(page: Page):
    """
    Test to verify the pipeline initialization and start/stop buttons.
    """
    page.goto(BASE_URL)

    init_button = page.locator("#initBtn")
    start_stop_button = page.locator("#startStopBtn")
    status_label = page.locator("#statusLabel")

    expect(init_button).to_be_visible()
    expect(start_stop_button).to_be_disabled()
    expect(status_label).to_have_text("Ready")
    print("UI Test: Initial state: Init button visible, Start/Stop disabled, Status: Ready.")

    # Click initialize button
    print("UI Test: Clicking 'Initialize Pipeline' button...")
    init_button.click()
    
    # Wait for models to initialize (status label should update)
    expect(status_label).to_have_text("Configuration updated and models re-initialized.", timeout=60000)
    expect(start_stop_button).not_to_be_disabled()
    print("UI Test: Models initialized, Start/Stop button enabled.")

    # Click start button
    print("UI Test: Clicking 'Start' button...")
    start_stop_button.click()
    expect(start_stop_button).to_have_text("Stop")
    expect(status_label).to_have_text("Listening...") # Assuming this is the next status
    print("UI Test: Pipeline started, Start/Stop button shows 'Stop', Status: Listening.")

    # Click stop button
    print("UI Test: Clicking 'Stop' button...")
    start_stop_button.click()
    expect(start_stop_button).to_have_text("Start")
    expect(status_label).to_have_text("Ready") # Assuming this is the status after stopping
    print("UI Test: Pipeline stopped, Start/Stop button shows 'Start', Status: Ready.")

@pytest.mark.ui
def test_language_and_tts_model_selection(page: Page):
    """
    Test to verify language selection and TTS model switching.
    """
    page.goto(BASE_URL)

    input_lang_select = page.locator("#inputLanguageSelect")
    output_lang_select = page.locator("#outputLanguageSelect")
    tts_model_select = page.locator("#ttsModelSelect")
    f5_voice_selection_group = page.locator("#f5-voice-selection-group")

    expect(input_lang_select).to_be_visible()
    expect(output_lang_select).to_be_visible()
    expect(tts_model_select).to_be_visible()
    expect(f5_voice_selection_group).to_be_hidden()
    print("UI Test: Language and TTS model selects visible. F5 voice selection hidden initially.")

    # Test input language selection
    input_lang_select.select_option("es")
    expect(input_lang_select).to_have_value("es")
    print("UI Test: Input language changed to Spanish.")

    # Test output language selection
    output_lang_select.select_option("de")
    expect(output_lang_select).to_have_value("de")
    print("UI Test: Output language changed to German.")

    # Test TTS model switch to F5
    tts_model_select.select_option("xtts")
    expect(tts_model_select).to_have_value("xtts")
    expect(f5_voice_selection_group).to_be_visible()
    print("UI Test: TTS model switched to F5, F5 voice selection group is visible.")

    # Test TTS model switch back to Piper
    tts_model_select.select_option("piper")
    expect(tts_model_select).to_have_value("piper")
    expect(f5_voice_selection_group).to_be_hidden()
    print("UI Test: TTS model switched back to Piper, F5 voice selection group is hidden.")

@pytest.mark.ui
def test_voice_recording_modal_interaction(page: Page):
    """
    Test to verify the voice recording modal opens, closes, and checkbox interaction.
    """
    page.goto(BASE_URL)

    record_voice_btn = page.locator("#recordVoiceBtn")
    voice_config_modal = page.locator("#voiceConfigModal")
    close_modal_btn = page.locator("#closeModalBtn")
    agree_checkbox = page.locator("#agreeToRecordCheckbox")
    start_recording_modal_btn = page.locator("#startRecordingModalBtn")
    upload_speaker_voice_modal_btn = page.locator("#uploadSpeakerVoiceModalBtn")

    expect(record_voice_btn).to_be_visible()
    expect(voice_config_modal).to_be_hidden()
    print("UI Test: Record Voice button visible, modal hidden initially.")

    # Open the modal
    print("UI Test: Clicking 'Record Voice' button to open modal...")
    record_voice_btn.click()
    expect(voice_config_modal).to_be_visible()
    expect(agree_checkbox).not_to_be_checked()
    expect(start_recording_modal_btn).to_be_disabled()
    expect(upload_speaker_voice_modal_btn).to_be_disabled()
    print("UI Test: Modal opened, checkbox unchecked, record/upload buttons disabled.")

    # Check the agreement checkbox
    print("UI Test: Checking agreement checkbox...")
    agree_checkbox.check()
    expect(agree_checkbox).to_be_checked()
    expect(start_recording_modal_btn).not_to_be_disabled()
    expect(upload_speaker_voice_modal_btn).not_to_be_disabled()
    print("UI Test: Checkbox checked, record/upload buttons enabled.")

    # Uncheck the agreement checkbox
    print("UI Test: Unchecking agreement checkbox...")
    agree_checkbox.uncheck()
    expect(agree_checkbox).not_to_be_checked()
    expect(start_recording_modal_btn).to_be_disabled()
    expect(upload_speaker_voice_modal_btn).to_be_disabled()
    print("UI Test: Checkbox unchecked, record/upload buttons disabled again.")

    # Check the agreement checkbox again to enable buttons for closing test
    agree_checkbox.check()

    # Close the modal
    print("UI Test: Clicking 'Close' button to close modal...")
    close_modal_btn.click()
    expect(voice_config_modal).to_be_hidden()
    print("UI Test: Modal closed.")

@pytest.mark.ui
def test_sidebar_and_account_button(page: Page):
    """
    Test to verify the sidebar opens and closes, and the account button is present.
    """
    page.goto(BASE_URL)

    burger_menu_btn = page.locator("#burgerMenuBtn")
    sidebar = page.locator("#sidebar")
    close_sidebar_btn = page.locator("#closeSidebarBtn")
    account_button = page.locator("#accountButton")

    expect(burger_menu_btn).to_be_visible()
    expect(sidebar).to_be_hidden()
    print("UI Test: Burger menu button visible, sidebar hidden initially.")

    # Open the sidebar
    print("UI Test: Clicking burger menu button to open sidebar...")
    burger_menu_btn.click()
    expect(sidebar).to_be_visible()
    expect(account_button).to_be_visible()
    print("UI Test: Sidebar opened, account button visible.")

    # Close the sidebar
    print("UI Test: Clicking close sidebar button...")
    close_sidebar_btn.click()
    expect(sidebar).to_be_hidden()
    print("UI Test: Sidebar closed.")
