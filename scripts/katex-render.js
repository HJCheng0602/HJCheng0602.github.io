// Server-side KaTeX rendering for Hexo + hexo-renderer-marked.
// hexo-renderer-marked converts math blocks into <br>$$<br>content<br>$$<br>
// or content $$</p> forms, and HTML-encodes special chars like = → &#x3D;
const katex = require('katex');

function decodeEntities(str) {
  return str
    .replace(/&#x3D;/g, '=')
    .replace(/&#x7C;/g, '|')
    .replace(/&#x2F;/g, '/')
    .replace(/&#x60;/g, '`')
    .replace(/&#xA0;/g, ' ')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&');
}

function renderMath(tex, displayMode) {
  try {
    return katex.renderToString(tex.trim(), {
      displayMode,
      throwOnError: false,
      output: 'html',
    });
  } catch (e) {
    return `<span class="katex-error" title="${e.message}">${tex}</span>`;
  }
}

// Let marked recognize math before it applies Markdown emphasis rules. Without
// this, a formula such as `_t ... _j` becomes `<em>t ... </em>j` before KaTeX
// sees it, and KaTeX renders the tag name as literal mathematical text.
hexo.extend.filter.register('marked:extensions', function(extensions) {
  extensions.push({
    name: 'blockMath',
    level: 'block',
    tokenizer(src) {
      const cap = /^\s{0,3}\$\$[ \t]*\n?([\s\S]*?)\n?[ \t]*\$\$(?:[ \t]*(?:\n|$))/.exec(src);
      if (!cap) return undefined;

      return {
        type: 'blockMath',
        raw: cap[0],
        math: cap[1],
      };
    },
    renderer(token) {
      return `<p>${renderMath(token.math, true)}</p>\n`;
    },
  });

  extensions.push({
    name: 'inlineMath',
    level: 'inline',
    start(src) {
      const index = src.indexOf('$');
      return index < 0 ? undefined : index;
    },
    tokenizer(src) {
      const cap = /^\$(?!\$)([^$\n]+?)\$(?!\$)/.exec(src);
      if (!cap) return undefined;

      return {
        type: 'inlineMath',
        raw: cap[0],
        math: cap[1],
      };
    },
    renderer(token) {
      return renderMath(token.math, false);
    },
  });
});

function restoreMarkdownEmphasis(tex) {
  return tex
    .replace(/<strong>([\s\S]*?)<\/strong>/g, '__$1__')
    .replace(/<em>([\s\S]*?)<\/em>/g, '_$1_');
}

hexo.extend.filter.register('after_render:html', function(html) {
  // Highlight.js splits CUDA <<<>>> into spaced tokens — merge them back inside code blocks
  html = html.replace(/<(?:pre|code)[^>]*>[\s\S]*?<\/(?:pre|code)>/g, function(block) {
    return block
      .replace(/(&lt;) (?=&lt;)/g, '$1')
      .replace(/(&gt;) (?=&gt;)/g, '$1');
  });

  // Display math: opening $$ is preceded by <br> or <p>
  //               closing $$ is followed by <br> or </p>
  html = html.replace(
    /(?:<br>|(?<=<p>))\s*\$\$([\s\S]*?)\$\$\s*(?=<br>|<\/p>)/g,
    function(_, inner) {
      // Strip <br> tags inside, decode HTML entities
      const tex = decodeEntities(restoreMarkdownEmphasis(inner).replace(/<br>/g, '\n'));
      return renderMath(tex, true);
    }
  );

  // Inline math: $...$ anywhere in the HTML
  // Guard: no newlines inside, no HTML tags (< excluded), not preceded/followed by $
  html = html.replace(
    /(?<!\$)\$(?!\$)([^$\n<]{1,300}?)(?<!\$)\$(?!\$)/g,
    function(_, tex) {
      return renderMath(decodeEntities(restoreMarkdownEmphasis(tex)), false);
    }
  );

  return html;
});
