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
  data.content = formatBody(data.content);
  return data;
});

function formatBody(text) {
  const lines = text.split('\n');
  const out = [];
  let inCode = false, inMath = false, blankRun = 0;

  for (const raw of lines) {
    if (/^(`{3,}|~{3,})/.test(raw)) { inCode = !inCode; blankRun = 0; out.push(raw); continue; }
    if (inCode) { out.push(raw); continue; }

    if (/^\s*\$\$/.test(raw)) { inMath = !inMath; blankRun = 0; out.push(raw); continue; }
    if (inMath) { out.push(raw); continue; }

    if (raw.trim() === '') {
      blankRun++;
      if (blankRun <= 2) out.push('');
      continue;
    }
    blankRun = 0;

    const hm = raw.match(/^(#{1,6} )(.+)$/);
    if (hm) {
      const t = hm[2].trimEnd();
      out.push(hm[1] + t.charAt(0).toUpperCase() + t.slice(1));
      continue;
    }

    out.push(formatProse(raw.trimEnd()));
  }

  return out.join('\n').replace(/\n*$/, '\n');
}

function formatProse(line) {
  const saved = [];
  const save = re => { line = line.replace(re, m => { saved.push(m); return `\x00${saved.length - 1}\x00`; }); };

  save(/`[^`]*`/g);
  save(/\$\$[\s\S]*?\$\$/g);
  save(/\$[^$\n]{1,200}?\$/g);
  save(/\[[^\]]*\]\([^)]*\)/g);
  save(/https?:\/\/\S+/g);

  line = line.replace(/ +([,\.;:!?])/g, '$1');
  line = line.replace(/,([^\s\x00\d,])/g,  ', $1');
  line = line.replace(/;([^\s\x00])/g,      '; $1');
  line = line.replace(/:([^\s\x00\/])/g,    ': $1');
  line = line.replace(/([a-z\)\]'"!?])\.([A-Z])/g, '$1. $2');

  const CJK = '一-鿿㐀-䶿＀-￯　-〿';
  line = line.replace(new RegExp(`([${CJK}])([A-Za-z0-9])`, 'g'), '$1 $2');
  line = line.replace(new RegExp(`([A-Za-z0-9])([${CJK}])`, 'g'), '$1 $2');

  line = line.replace(/(\S) {2,}/g, '$1 ');

  return line.replace(/\x00(\d+)\x00/g, (_, i) => saved[+i]);
}
