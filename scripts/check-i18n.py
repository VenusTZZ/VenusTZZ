#!/usr/bin/env python3
"""检查交互页英文副本的结构与漏翻文案。

用法：
    python3 scripts/check-i18n.py docs/public/notes
    python3 scripts/check-i18n.py docs/public/notes/npu-quant-comm.html
"""
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

CJK = re.compile(r'[\u4e00-\u9fff]')
STRUCT_ATTRS = {'id', 'class', 'href', 'src'}
TEXT_ATTRS = {'title', 'alt', 'aria-label', 'placeholder'}


class Skeleton(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.items = []
        self.texts = []
        self.attrs = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self._skip += 1
        normalized = sorted(
            (name, value if name in STRUCT_ATTRS else None) for name, value in attrs
        )
        self.items.append(('start', tag, tuple(normalized)))
        for name, value in attrs:
            if name in TEXT_ATTRS and value and CJK.search(value):
                self.attrs.append((tag, name, value.strip()))

    def handle_startendtag(self, tag, attrs):
        normalized = sorted(
            (name, value if name in STRUCT_ATTRS else None) for name, value in attrs
        )
        self.items.append(('startend', tag, tuple(normalized)))

    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self._skip:
            self._skip -= 1
        self.items.append(('end', tag, ()))

    def handle_data(self, data):
        if self._skip:
            return
        text = data.strip()
        if text and CJK.search(text):
            self.texts.append(text)


def script_text(html):
    return '\n'.join(
        re.findall(r'<script\b(?![^>]*\bsrc=)[^>]*>(.*?)</script>', html, re.S | re.I)
    )


def js_cjk_strings(html):
    js = script_text(html)
    found = []
    for match in re.finditer(r"(['\"`])((?:\\.|(?!\1).)*?)\1", js, re.S):
        value = match.group(2)
        if CJK.search(value):
            found.append(value.strip())
    return found


def strip_lang_switch(html):
    html = re.sub(r'<!-- lang-switch:start -->[\s\S]*?<!-- lang-switch:end -->\s*', '', html)
    html = html.replace('<link rel="stylesheet" href="./lang-switch.css">\n', '')
    return html


def compare(base_path, en_path):
    base_html = base_path.read_text(encoding='utf-8')
    en_html = en_path.read_text(encoding='utf-8')
    has_switch = 'class="lang-switch"' in en_html and 'data-lang="zh"' in en_html
    base_html = strip_lang_switch(base_html)
    en_html = strip_lang_switch(en_html)

    base = Skeleton()
    base.feed(base_html)
    en = Skeleton()
    en.feed(en_html)

    problems = []
    if base.items != en.items:
        # 找出第一处结构差异
        for index, (a, b) in enumerate(zip(base.items, en.items)):
            if a != b:
                problems.append(f'结构差异 @{index}: {a} != {b}')
                break
        if len(base.items) != len(en.items):
            problems.append(f'标签数量不同: zh={len(base.items)} en={len(en.items)}')

    if en.texts:
        problems.append(f'英文页仍有 {len(en.texts)} 处中文文本节点')
    if en.attrs:
        problems.append(f'英文页仍有 {len(en.attrs)} 处中文属性')
    remaining_js = js_cjk_strings(en_html)
    if remaining_js:
        problems.append(f'英文页 JS 仍有 {len(remaining_js)} 个中文字符串')
    if not has_switch:
        problems.append('英文页缺少注入的中文切换按钮')

    print(f'\n{base_path.name} -> {en_path.name}')
    if problems:
        for problem in problems:
            print(f'  ✗ {problem}')
        for item in base.texts[:5]:
            print(f'    zh text sample: {item[:80]}')
        for item in en.texts[:8]:
            print(f'    en text: {item[:80]}')
        for item in en.attrs[:8]:
            print(f'    en attr: {item}')
        for item in remaining_js[:8]:
            print(f'    en js: {item[:80]}')
        return False

    print(f'  ✓ skeleton identical, zh texts={len(base.texts)}, JS strings clean, switch present')
    return True


def main(target):
    root = Path(target)
    if root.is_file():
        pairs = [(root, root.with_name(root.stem + '.en.html'))]
    else:
        pairs = [
            (path, path.with_name(path.stem + '.en.html'))
            for path in sorted(root.glob('*.html'))
            if not path.name.endswith('.en.html')
        ]

    ok = True
    pending = []
    for base, en in pairs:
        if not en.exists():
            pending.append(base.name)
            continue
        ok = compare(base, en) and ok

    if pending:
        print('\n尚未生成英文版：')
        for name in pending:
            print(f'  - {name}')

    return 0 if ok and not pending else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'docs/public/notes'))
