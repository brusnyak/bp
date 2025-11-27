import pytest
from playwright.sync_api import Page, expect

# Assuming the app is running on localhost:8000 with HTTPS
BASE_URL = "https://localhost:8000/ui/live-speech/live.html"

@pytest.mark.ui
def test_latency_chart_exists(page: Page):
    """
    Test to ensure the latency timeline chart canvas is present on the page.
    """
    page.goto(BASE_URL)

    # Expect the chart canvas to be visible
    chart_canvas = page.locator("#latencyTimelineChart")
    expect(chart_canvas).to_be_visible()
    print(f"UI Test: Latency timeline chart canvas found and is visible on {BASE_URL}")

# You can add more tests here to interact with the chart,
# e.g., checking if data is rendered, if it updates, etc.
