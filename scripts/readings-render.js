'use strict';

function hasCategory(data, name) {
  if (!data.categories) return false;
  // after_post_render: categories is a Warehouse QueryResult
  if (typeof data.categories.toArray === 'function') {
    return data.categories.toArray().some(c => c.name === name);
  }
  return [].concat(data.categories).flat(Infinity).includes(name);
}

// Minimal BibTeX parser — handles nested braces and {{double-braced}} titles
function parseBibtex(src) {
  const out = {};

  const typeM = src.match(/@(\w+)\s*\{/);
  if (typeM) out.type = typeM[1].toLowerCase();

  let i = src.indexOf('{');
  if (i < 0) return out;
  i++;

  const commaIdx = src.indexOf(',', i);
  if (commaIdx < 0) return out;
  out.key = src.slice(i, commaIdx).trim();
  i = commaIdx + 1;

  while (i < src.length) {
    while (i < src.length && /\s/.test(src[i])) i++;
    if (i >= src.length || src[i] === '}') break;

    const nameStart = i;
    while (i < src.length && /\w/.test(src[i])) i++;
    const fieldName = src.slice(nameStart, i).toLowerCase();
    if (!fieldName) { i++; continue; }

    while (i < src.length && src[i] !== '=') i++;
    i++;
    while (i < src.length && /\s/.test(src[i])) i++;

    let value = '';
    if (src[i] === '{') {
      let depth = 0;
      i++;
      const valStart = i;
      while (i < src.length) {
        if (src[i] === '{') { depth++; i++; }
        else if (src[i] === '}') {
          if (depth === 0) { i++; break; }
          depth--; i++;
        } else { i++; }
      }
      value = src.slice(valStart, i - 1)
        .replace(/^\{|\}$/g, '')  // strip one layer of {{...}}
        .replace(/[{}]/g, '')     // strip any remaining braces
        .trim();
    } else if (src[i] === '"') {
      i++;
      const valStart = i;
      while (i < src.length && src[i] !== '"') i++;
      value = src.slice(valStart, i).trim();
      i++;
    } else {
      const valStart = i;
      while (i < src.length && src[i] !== ',' && src[i] !== '}') i++;
      value = src.slice(valStart, i).trim();
    }

    if (fieldName && value) out[fieldName] = value;
    while (i < src.length && (src[i] === ',' || /\s/.test(src[i]))) i++;
  }

  return out;
}

function formatAuthors(authorStr) {
  const authors = authorStr.split(/\s+and\s+/i).map(a => {
    a = a.trim();
    const parts = a.split(',');
    if (parts.length === 2) return `${parts[1].trim()} ${parts[0].trim()}`;
    return a;
  });
  if (authors.length > 3) return authors[0] + ' et al.';
  return authors.join(', ');
}

function getVenue(f) {
  if (f.journal)       return f.journal;
  if (f.booktitle)     return f.booktitle;
  if (f.series)        return f.series;
  if (f.publisher)     return f.publisher;
  if (f.howpublished)  return f.howpublished;
  if (f.url) {
    try { return new URL(f.url).hostname.replace(/^www\./, ''); } catch (e) {}
  }
  return '';
}

hexo.extend.filter.register('after_post_render', function(data) {
  if (!hasCategory(data, 'readings')) return data;

  const bibtexRaw = data.bibtex;
  if (bibtexRaw && String(bibtexRaw).trim()) {
    const f = parseBibtex(String(bibtexRaw));

    const venue  = [getVenue(f), f.year].filter(Boolean).join(' · ');
    const title  = f.title || '';
    const authors = f.author ? formatAuthors(f.author) : '';
    const url    = f.url || '';

    const links = [];
    if (url) links.push(`<a href="${url}" target="_blank" rel="noopener"><i class="fas fa-file-alt"></i> Paper</a>`);
    if (f.code) links.push(`<a href="${f.code}" target="_blank" rel="noopener"><i class="fab fa-github"></i> Code</a>`);

    if (title) {
      const card = `<div class="paper-card">
${venue  ? `<div class="paper-venue">${venue}</div>` : ''}
<div class="paper-title">${title}</div>
${authors ? `<div class="paper-authors">${authors}</div>` : ''}
${links.length ? `<div class="paper-links">${links.join('')}</div>` : ''}
</div>

`;
      data.content = card + data.content;
    }
  }

  const refs = [].concat(data.references || []).filter(r => r && String(r).trim());
  if (refs.length) {
    data.content += `\n\n<div class="post-references">
<h2 class="references-heading">References</h2>
<ol>
${refs.map(r => `<li>${r}</li>`).join('\n')}
</ol>
</div>`;
  }

  return data;
});
