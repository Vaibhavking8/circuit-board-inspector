"""Quick sanity check for index.html and styles.css."""
import pathlib, sys

root = pathlib.Path(__file__).parent.parent
html = (root / "static" / "index.html").read_text(encoding="utf-8")
css  = (root / "static" / "css" / "styles.css").read_text(encoding="utf-8")

errors = []

if "https://" in html:   errors.append("Found https:// in HTML")
if "http://" in html:    errors.append("Found http:// in HTML")
if "http" in css:        errors.append("Found 'http' in CSS")
if 'type="module"' not in html: errors.append("Missing type=module on script tag")
for id_ in ("step-reference", "step-annotate", "step-inspect"):
    if id_ not in html: errors.append(f"Missing id={id_} in HTML")
for token in ("--color-ok:", "--color-missing:", "--color-uncertain:"):
    if token not in css: errors.append(f"Missing CSS token {token}")

if errors:
    print("FAILURES:")
    for e in errors:
        print(" ", e)
    sys.exit(1)

print(f"HTML: {len(html)} chars")
print(f"CSS:  {len(css)} chars")
print("index.html and styles.css OK")
