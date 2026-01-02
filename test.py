import os
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

def main(web_path, output):

    html_path = Path(web_path).expanduser().resolve()
    if not html_path.exists():
        print(f"文件不存在: {html_path}")
        sys.exit(1)

    out_path = Path(output).expanduser().resolve()

    # 生成 file:// URL（Mac/Linux 直接用 as_uri 即可）
    url = html_path.as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1280, "height": 720},
            device_scale_factor=2,
        )

        page.goto(url, wait_until="load")
        page.wait_for_timeout(200)  # 可选：等渲染更稳

        page.screenshot(
            path=str(out_path),
            full_page=True,           # 关键：整页长图
            animations="disabled",
        )
        browser.close()

    print(str(out_path))

if __name__ == "__main__":
    main("/Users/pedestrian/Desktop/web_case/results/gpt-5-codex/text/edit/2953746_www.vanopstal.be_L3_2/ans/index.html", "/Users/pedestrian/Desktop/web_case/results/gpt-5-codex/text/edit/2953746_www.vanopstal.be_L3_2/ans/screenshot_index.png")
    