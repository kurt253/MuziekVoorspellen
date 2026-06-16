"""Maak screenshots van de Streamlit-app voor de presentatie.

Gebruik:
    python scripts/maak_screenshots.py
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, Page

ROOT    = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "presentatie_screenshots"
IMG_DIR.mkdir(exist_ok=True)

APP_URL  = "http://localhost:8501"
GENEREER = APP_URL + "/Genereer"
TRAINING = APP_URL + "/Training"


def wacht_op_app(page: Page) -> None:
    page.wait_for_selector('[data-testid="stApp"]', timeout=90_000)
    time.sleep(4)


def screenshot(page: Page, naam: str) -> None:
    pad = IMG_DIR / naam
    page.screenshot(path=str(pad), full_page=False)
    size = pad.stat().st_size // 1024
    print(f"  Opgeslagen: {pad.name} ({size} KB)")


def screenshot_overzicht(page: Page) -> None:
    print(">> Overzichtspagina...")
    page.goto(APP_URL)
    wacht_op_app(page)
    screenshot(page, "scherm_overzicht.png")


def screenshot_training(page: Page) -> None:
    print(">> Trainingspagina...")
    page.goto(TRAINING)
    wacht_op_app(page)
    screenshot(page, "scherm_training.png")


def screenshot_genereer_leeg(page: Page) -> None:
    print(">> Genereerpagina (leeg)...")
    page.goto(GENEREER)
    wacht_op_app(page)
    screenshot(page, "scherm_genereer.png")


def stel_filter_in(page: Page, filter_tekst: str) -> None:
    try:
        # Tweede selectbox = genre-filter
        selectboxen = page.locator('[data-testid="stSelectbox"]')
        if selectboxen.count() < 2:
            return
        selectboxen.nth(1).click()
        time.sleep(0.5)
        optie = page.locator(
            '[data-testid="stSelectboxVirtualDropdown"] li'
        ).filter(has_text=filter_tekst).first
        optie.wait_for(state="visible", timeout=5_000)
        optie.click()
        time.sleep(2)
        print(f"   Filter '{filter_tekst}' ingesteld.")
    except Exception as e:
        print(f"   [filter overgeslagen: {str(e)[:120]}]")


def klik_rij(page: Page, rij_idx: int) -> bool:
    """
    Klik data-rij rij_idx (0-gebaseerd) in het Streamlit Glide Data Grid.
    Streamlit 1.x gebruikt een canvas-gebaseerde tabel — geen klikbare DOM-elementen.
    We berekenen de viewport-positie van de canvas en klikken op de juiste y.
    """
    ROW_HEIGHT   = 35   # Glide Data Grid standaard rijhoogte (px)
    HEADER_H     = 36   # hoogte van de headerrij
    CLICK_OFFSET = ROW_HEIGHT * rij_idx + HEADER_H + ROW_HEIGHT // 2

    # Scroll de dataframe in beeld en haal zijn positie op
    try:
        info = page.evaluate("""
            (function() {
                const el = document.querySelector('[data-testid="stDataFrame"]');
                if (!el) return null;
                el.scrollIntoView({behavior: 'instant', block: 'start'});
                const rect = el.getBoundingClientRect();
                // Zoek ook canvas erin
                const canvas = el.querySelector('canvas');
                const cRect  = canvas ? canvas.getBoundingClientRect() : null;
                return {
                    el: {x: rect.left, y: rect.top, w: rect.width, h: rect.height},
                    cv: cRect ? {x: cRect.left, y: cRect.top, w: cRect.width, h: cRect.height} : null
                };
            })()
        """)
        print(f"   DataFame bbox: {info}")
        time.sleep(0.5)

        if info:
            # Gebruik de canvas als die gevonden is, anders de container
            src = info.get('cv') or info.get('el')
            if src and src.get('w', 0) > 0:
                cx = src['x'] + src['w'] / 2
                cy = src['y'] + CLICK_OFFSET
                if 0 <= cy <= 1200:
                    page.mouse.click(cx, cy)
                    time.sleep(2)
                    if page.locator("text=Geselecteerd:").count() > 0:
                        print(f"   Canvas-klik gelukt ({cx:.0f}, {cy:.0f}).")
                        return True
                    print(f"   Canvas-klik gedaan maar geen selectie. pos=({cx:.0f},{cy:.0f})")
    except Exception as e:
        print(f"   [canvas-klik mislukt: {str(e)[:120]}]")

    # Fallback: probeer het middelste punt van de volledige stDataFrame div
    try:
        box = page.locator('[data-testid="stDataFrame"]').first.bounding_box()
        if box:
            cx = box['x'] + box['width'] / 2
            cy = box['y'] + HEADER_H + ROW_HEIGHT * rij_idx + ROW_HEIGHT / 2
            page.mouse.click(cx, cy)
            time.sleep(2)
            if page.locator("text=Geselecteerd:").count() > 0:
                print(f"   Bounding-box klik gelukt ({cx:.0f},{cy:.0f}).")
                return True
            print(f"   Bounding-box klik zonder selectie ({cx:.0f},{cy:.0f}).")
    except Exception as e:
        print(f"   [bounding-box fallback mislukt: {str(e)[:100]}]")

    return False


def klik_genereer(page: Page) -> bool:
    for selector in [
        'button:has-text("Genereer aanvulling")',
        'button:has-text("Genereer")',
    ]:
        try:
            btn = page.locator(selector).first
            btn.scroll_into_view_if_needed()
            btn.wait_for(state="visible", timeout=6_000)
            btn.click()
            print("   Genereer-knop geklikt.")
            return True
        except Exception:
            pass
    print("   [genereer-knop niet gevonden]")
    return False


def wacht_op_midi(page: Page) -> None:
    try:
        page.wait_for_selector("midi-player", timeout=180_000)
        time.sleep(4)
        print("   Piano-roll geladen.")
    except Exception:
        print("   [piano-roll niet gezien, wacht 8s]")
        time.sleep(8)


def genereer_en_screenshot(page: Page, rij: int, naam: str,
                            genre: str = "cantate") -> None:
    print(f">> Genereren {naam} (rij={rij}, genre={genre})...")
    page.goto(GENEREER)
    wacht_op_app(page)

    stel_filter_in(page, genre)

    # Debug: hoeveel rows zijn er?
    n_rows = page.locator('[role="row"]').count()
    print(f"   Gevonden [role=row] elementen: {n_rows}")

    if not klik_rij(page, rij):
        print("   [rij selectie mislukt — screenshot zonder generatie]")
        screenshot(page, naam)
        return

    try:
        page.wait_for_selector("text=Geselecteerd:", timeout=6_000)
        print("   Selectie bevestigd.")
    except Exception:
        print("   [selectie niet bevestigd]")

    klik_genereer(page)
    wacht_op_midi(page)

    # Scroll naar boven voor screenshot
    page.evaluate("window.scrollTo(0, 0)")
    time.sleep(0.5)
    screenshot(page, naam)


def main() -> None:
    print("Streamlit starten...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
         "--server.headless", "true",
         "--server.port", "8501",
         "--browser.gatherUsageStats", "false"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(ROOT),
    )

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx  = browser.new_context(viewport={"width": 1600, "height": 1200})
            page = ctx.new_page()

            print("Wachten op Streamlit (30s)...")
            time.sleep(30)

            screenshot_overzicht(page)
            screenshot_training(page)
            screenshot_genereer_leeg(page)

            genereer_en_screenshot(page, rij=0, naam="voorbeeld_1.png", genre="cantate")
            genereer_en_screenshot(page, rij=2, naam="voorbeeld_2.png", genre="cantate")
            genereer_en_screenshot(page, rij=4, naam="voorbeeld_3.png", genre="cantate")

            browser.close()
    finally:
        proc.terminate()
        print("Streamlit gestopt.")

    print(f"\nScreenshots in: {IMG_DIR}")
    for naam in ["voorbeeld_1.png", "voorbeeld_2.png", "voorbeeld_3.png"]:
        p = IMG_DIR / naam
        status = f"OK ({p.stat().st_size // 1024} KB)" if p.exists() else "ontbreekt"
        print(f"  {naam}: {status}")


if __name__ == "__main__":
    main()
