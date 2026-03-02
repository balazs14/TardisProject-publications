#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


NORMAL_HEADER = """\\usepackage{natbib}
\\bibliographystyle{agsm}
"""

DRAFT_HEADER = """% TEMP editing mode: disable bibliography/citation processing while drafting
%\\usepackage{natbib}
%\\bibliographystyle{agsm}
\\usepackage{xparse}
\\RenewDocumentCommand{\\cite}{o m}{}
\\NewDocumentCommand{\\citep}{o o m}{}
\\NewDocumentCommand{\\citet}{o o m}{}
\\NewDocumentCommand{\\citeauthor}{o m}{}
\\NewDocumentCommand{\\citeyear}{o m}{}
\\AtBeginDocument{%
    \\RenewDocumentCommand{\\ref}{m}{}%
    \\RenewDocumentCommand{\\pageref}{m}{}%
    \\RenewDocumentCommand{\\eqref}{m}{}%
}
"""

NORMAL_BIB_FOOTER = """\\bibliography{bibl}
\\bibliographystyle{vancouver}
"""

DRAFT_BIB_FOOTER = """% TEMP editing mode
%\\bibliography{bibl}
%\\bibliographystyle{vancouver}
"""


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python publications/disable_draft_mode.py path/to/file.tex")
        return 1

    tex_path = Path(sys.argv[1])
    content = tex_path.read_text(encoding="utf-8")
    original = content

    content = content.replace(DRAFT_HEADER, NORMAL_HEADER, 1)
    content = content.replace(DRAFT_BIB_FOOTER, NORMAL_BIB_FOOTER, 1)

    if content != original:
        tex_path.write_text(content, encoding="utf-8")
        print(f"Disabled draft mode in {tex_path}")
    else:
        print(f"No changes needed for {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
