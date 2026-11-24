"""Validate a closed table fragment and normalize it for upstream TEDS.

TEDS expects html/body/table and handles td rather than th content. The same
normalization is applied to targets and predictions. No damaged HTML is repaired.
"""

import html as std_html
import re
from html.parser import HTMLParser

from lxml import etree, html


NORMALIZATION_VERSION = 'table-html-v2-data-bbox'


class InvalidTable(ValueError):
    """Prediction or annotation is not a complete, supported table."""


class _BalancedHTML(HTMLParser):
    ALLOWED = {'html', 'head', 'body', 'table', 'thead', 'tbody', 'tfoot', 'tr',
               'td', 'th', 'b', 'i', 'sup', 'sub', 'strong', 'em', 'span', 'br'}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.table_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in self.ALLOWED:
            raise InvalidTable(f'Unsupported tag: {tag}')
        structural = {'thead', 'tbody', 'tfoot', 'tr', 'td', 'th'}
        valid_parents = {'thead': {'table'}, 'tbody': {'table'}, 'tfoot': {'table'},
                         'tr': {'table', 'thead', 'tbody', 'tfoot'}, 'td': {'tr'}, 'th': {'tr'}}
        if tag in structural and (not self.stack or self.stack[-1] not in valid_parents[tag]):
            raise InvalidTable(f'Invalid parent of {tag}')
        if tag in {'b', 'i', 'sup', 'sub', 'strong', 'em', 'span', 'br'} and not ({'td', 'th'} & set(self.stack)):
            raise InvalidTable('Inline formatting outside table cell')
        if tag == 'table':
            self.table_count += 1
            if self.table_count != 1:
                raise InvalidTable('Expected exactly one table')
        if tag != 'br':
            self.stack.append(tag)
        if len({name for name, value in attrs}) != len(attrs):
            raise InvalidTable('Duplicate HTML attributes')
        for name, value in attrs:
            if name in {'rowspan', 'colspan'}:
                if not value or not value.isdigit() or not 1 <= int(value) <= 1000:
                    raise InvalidTable(f'Invalid {name}')
            elif name not in {'border', 'class', 'id', 'style', 'width', 'height', 'align',
                              'valign', 'cellspacing', 'cellpadding', 'rules', 'frame', 'data-bbox'}:
                raise InvalidTable(f'Unsupported attribute: {name}')

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag != 'br':
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack[-1] != tag:
            raise InvalidTable(f'Unbalanced closing tag: {tag}')
        self.stack.pop()

    def handle_data(self, data: str) -> None:
        if data.strip() and not ({'td', 'th'} & set(self.stack)):
            raise InvalidTable('Text outside table cell')


def canonical_table(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidTable('Empty output')
    value = value.strip()
    fence = re.fullmatch(r'```(?:html)?\s*\n(.*?)\n```', value, flags=re.S | re.I)
    if fence:
        value = fence.group(1).strip()
    if len(value) > 2_000_000:
        raise InvalidTable('Table exceeds parser limit')
    checker = _BalancedHTML()
    checker.feed(value)
    checker.close()
    if checker.stack or checker.table_count != 1:
        raise InvalidTable('Incomplete table')
    doc = html.document_fromstring(value, parser=html.HTMLParser(no_network=True))
    tables = doc.xpath('//table')
    if len(tables) != 1 or tables[0].xpath('ancestor::table'):
        raise InvalidTable('Expected one nonnested table')
    table = tables[0]
    for node in table.iter():
        if node.tag == 'th':
            node.tag = 'td'
        attrs = {k: v for k, v in node.attrib.items() if k in {'rowspan', 'colspan'}}
        node.attrib.clear()
        node.attrib.update(attrs)
    if not table.xpath('.//tr') or not table.xpath('.//td'):
        raise InvalidTable('Table has no rows/cells')
    # Enforce table structure rather than letting lxml silently repair bad output.
    for node in table.iter():
        allowed = {'table': {'thead', 'tbody', 'tfoot', 'tr'},
                   'thead': {'tr'}, 'tbody': {'tr'}, 'tfoot': {'tr'}, 'tr': {'td'}}
        if node.tag in allowed and any(c.tag not in allowed[node.tag] for c in node):
            raise InvalidTable(f'Invalid child of {node.tag}')
        if node.tag == 'td' and any(c.tag in {'table', 'tr', 'td'} for c in node.iterdescendants()):
            raise InvalidTable('Nested structural element inside cell')
    fragment = etree.tostring(table, encoding='unicode', method='html', with_tail=False)
    return '<html><body>' + fragment + '</body></html>'


def annotation_html(row: dict) -> str:
    """Assemble PubTabNet structure/cells using its documented token schema.

    Follows the upstream examples/utils.py assembly idea; cell count is checked,
    and literal characters are escaped without losing allowed inline markup.
    """
    annotation = row['html']
    structure = ''.join(annotation['structure']['tokens'])
    cells = annotation['cells']
    count = 0

    def insert(match: re.Match) -> str:
        nonlocal count
        if count >= len(cells):
            raise InvalidTable('More structure cells than cell annotations')
        tokens = cells[count]['tokens']
        if not all(isinstance(token, str) for token in tokens):
            raise InvalidTable('Non-string cell token')
        text = ''.join(token if re.fullmatch(r'</?(?:b|i|sup|sub|strong|em)>', token)
                       else std_html.escape(token, quote=False) for token in tokens)
        count += 1
        return match.group(1) + text + '</td>'

    filled = re.sub(r'(<td\b[^<>]*>)(</td>)', insert, structure)
    if count != len(cells):
        raise InvalidTable('Cell count does not match structure')
    return canonical_table('<table>' + filled + '</table>')


def cell_texts(value: str) -> list[str]:
    doc = html.document_fromstring(value)
    return [' '.join(cell.text_content().split()) for cell in doc.xpath('//table//td')]


def table_attributes(value: str) -> dict:
    doc = html.document_fromstring(value)
    cells = doc.xpath('//table//td')
    return {'rows': len(doc.xpath('//table//tr')), 'cells': len(cells),
            'merged': any(int(c.get('rowspan', '1')) > 1 or int(c.get('colspan', '1')) > 1 for c in cells),
            'numeric_dense': sum(bool(re.search(r'\d', c.text_content())) for c in cells) >= len(cells) / 2}

