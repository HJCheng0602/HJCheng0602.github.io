'use strict';

hexo.extend.filter.register('before_post_render', function(data) {
  // Rewrite relative <img src> to absolute paths so assets resolve on tag/archive pages
  if (data.permalink) {
    const base = data.permalink.replace(/\/?$/, '/');
    data.content = data.content.replace(
      /(<img\s[^>]*src=")(?!https?:\/\/|\/|data:)([^"]+)(")/g,
      (_, pre, src, post) => `${pre}${base}${src}${post}`
    );
  }
  return data;
});

// Typora exports often place a `$$` block directly under a prose line, sometimes
// with the formula starting on the opening line (`$$\pi(...)`) or with both
// delimiters on one line (`$$ a = b $$`). marked then parses them as part of the
// surrounding paragraph: the blockMath extension in scripts/katex-render.js never
// fires and rendering falls back to a lossy after_render regex (backslash escapes
// already collapsed, `=` lines turn into setext headings). Pad top-level `$$`
// delimiters with blank lines so each block reaches KaTeX as raw TeX. Only
// column-0 delimiters are tracked — indented ones belong to nested markdown
// structures (lists, blockquotes) and are left untouched.
hexo.extend.filter.register('before_post_render', function(data) {
  const lines = data.content.split('\n');
  const out = [];
  let inCode = false;
  let inMath = false;
  let closed = false;

  for (const line of lines) {
    if (/^\s*(```|~~~)/.test(line)) inCode = !inCode;

    const delim = !inCode && /^\$\$/.test(line);
    const lineOnly = delim && /^\$\$[ \t]*$/.test(line);
    const oneLiner = delim && !lineOnly && /\$\$[ \t]*$/.test(line);

    if (delim && !inMath) {
      if (out.length && out[out.length - 1].trim() !== '') out.push('');
      inMath = !lineOnly && !oneLiner;
      closed = oneLiner;
    } else if (delim && inMath && lineOnly) {
      inMath = false;
      closed = true;
    } else if (closed) {
      closed = false;
      if (line.trim() !== '') out.push('');
    }

    out.push(line);
  }

  data.content = out.join('\n');
  return data;
});
