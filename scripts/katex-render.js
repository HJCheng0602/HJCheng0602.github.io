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

hexo.extend.filter.register('after_render:html', function(html) {
  // Display math: opening $$ is preceded by <br> or <p>
  //               closing $$ is followed by <br> or </p>
  html = html.replace(
    /(?:<br>|(?<=<p>))\s*\$\$([\s\S]*?)\$\$\s*(?=<br>|<\/p>)/g,
    function(_, inner) {
      // Strip <br> tags inside, decode HTML entities
      const tex = decodeEntities(inner.replace(/<br>/g, '\n'));
      return renderMath(tex, true);
    }
  );

  // Inline math: $...$ within article content divs
  // Guard: no newlines inside, not empty, not preceded/followed by $
  html = html.replace(
    /(<div class="content">[\s\S]*?<\/div>)/g,
    function(block) {
      return block.replace(
        /(?<!\$)\$(?!\$)([^$\n<]{1,300}?)(?<!\$)\$(?!\$)/g,
        function(_, tex) {
          return renderMath(decodeEntities(tex), false);
        }
      );
    }
  );

  return html;
});
