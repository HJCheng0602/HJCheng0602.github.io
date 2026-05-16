// Append a citation block to the end of every post page.
hexo.extend.filter.register('after_render:html', function(html, data) {
  const isPost = data && data.path && /^blog\/[^/]+\/index\.html$/.test(data.path);
  if (!isPost) return html;

  const siteUrl = hexo.config.url.replace(/\/$/, '');
  // Strip trailing index.html so URL ends with /
  const postUrl = siteUrl + '/' + data.path.replace(/index\.html$/, '');

  // Extract title: strip " - Site Name" suffix
  const titleMatch = html.match(/<title>([^<]+)<\/title>/);
  const rawTitle = titleMatch ? titleMatch[1].trim() : '';
  const title = rawTitle.replace(/\s*[-|].*$/, '').trim()
    .replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // Extract dates from article-meta section (camelCase dateTime attribute)
  const metaMatch = html.match(/article-meta[^]*?<\/div>/);
  function fmtDate(iso) {
    if (!iso) return '—';
    const dt = new Date(iso);
    const mm = String(dt.getMonth() + 1).padStart(2, '0');
    const dd = String(dt.getDate()).padStart(2, '0');
    const yyyy = dt.getFullYear();
    return `${mm}-${dd}-${yyyy}`;
  }

  let posted = '—', updated = '—';
  if (metaMatch) {
    const metaHtml = metaMatch[0];
    const times = [...metaHtml.matchAll(/dateTime="([^"]+)"/g)];
    if (times[0]) posted  = fmtDate(times[0][1]);
    if (times[1]) updated = fmtDate(times[1][1]);
    else updated = posted;
  }

  const block = `<div class="post-citation">
  <p class="citation-title">${title}</p>
  <p class="citation-url"><a href="${postUrl}">${postUrl}</a></p>
  <div class="citation-meta">
    <div class="citation-col"><span class="citation-label">Author</span><span class="citation-value">Jincheng Han</span></div>
    <div class="citation-col"><span class="citation-label">Posted on</span><span class="citation-value">${posted}</span></div>
    <div class="citation-col"><span class="citation-label">Updated on</span><span class="citation-value">${updated}</span></div>
  </div>
</div>`;

  // Insert inside the article, just before the tags line
  const tagsAnchor = '<div class="article-tags';
  if (html.includes(tagsAnchor)) {
    return html.replace(tagsAnchor, block + '\n<div class="article-tags');
  }
  return html.replace('</article>', block + '</article>');
});
