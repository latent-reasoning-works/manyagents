"""The example harness tolerates editorial changes, but rejects ambiguity."""

import shlex

import pytest


def test_example_markers_ignore_headings_and_shell_layout(documented_example):
    markdown = '''## arbitrary heading

<!-- example:sweep -->
```bash
manyagents --multirun \\
  experiment=invariance_full
```
<!-- /example:sweep -->

Unrelated prose and `manyagents --multirun` mentions.
'''
    assert shlex.split(documented_example(markdown, 'sweep', 'bash')) == [
        'manyagents', '--multirun', 'experiment=invariance_full',
    ]


@pytest.mark.parametrize('markdown', [
    '',
    '<!-- example:sweep --><!-- /example:sweep -->',
    '<!-- example:sweep -->```python\npass\n```<!-- /example:sweep -->',
    '<!-- example:sweep --><!-- example:sweep --><!-- /example:sweep -->',
])
def test_invalid_example_fails_clearly(markdown, documented_example):
    with pytest.raises(AssertionError):
        documented_example(markdown, 'sweep', 'bash')
