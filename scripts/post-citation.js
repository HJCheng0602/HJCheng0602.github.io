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

  const metaMatch = html.match(/article-meta[^]*?<\/div>/);

  // article-meta.js already replaced <time dateTime="..."> with formatted MM-DD-YYYY strings,
  // so just grab those directly from the meta section.
  let posted = '—', updated = '—';
  if (metaMatch) {
    const dates = [...metaMatch[0].matchAll(/(\d{2}-\d{2}-\d{4})/g)];
    if (dates[0]) posted  = dates[0][1];
    if (dates[1]) updated = dates[1][1];
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
