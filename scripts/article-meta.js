// Rewrite article-meta on post pages: add FA icons, reformat dates to MM-DD-YYYY,
// strip verbose "Posted"/"Updated" text labels.
hexo.extend.filter.register('after_render:html', function(html, data) {
  const isPost = data && data.path && /^blog\/[^/]+\/index\.html$/.test(data.path);
  if (!isPost) return html;

  function fmtDate(iso) {
    const dt = new Date(iso);
    const mm = String(dt.getMonth() + 1).padStart(2, '0');
    const dd = String(dt.getDate()).padStart(2, '0');
    return `${mm}-${dd}-${dt.getFullYear()}`;
  }

  // Reformat Posted date span
  html = html.replace(
    /<span class="level-item">Posted&nbsp;<time dateTime="([^"]+)"[^>]*>[^<]*<\/time><\/span>/,
    (_, iso) => `<span class="level-item"><i class="fas fa-calendar-alt meta-icon"></i>${fmtDate(iso)}</span>`
  );

  // Reformat Updated date span
  html = html.replace(
    /<span class="level-item">Updated&nbsp;<time dateTime="([^"]+)"[^>]*>[^<]*<\/time><\/span>/,
    (_, iso) => `<span class="level-item"><i class="fas fa-calendar-check meta-icon"></i>${fmtDate(iso)}</span>`
  );

  // Add folder icon before category link
  html = html.replace(
    /(<span class="level-item">)(<a class="link-muted" href="\/categories\/)/,
    '$1<i class="fas fa-folder meta-icon"></i>$2'
  );

  // Add clock icon before read time
  html = html.replace(
    /(<span class="level-item">)(\d+ minutes? read)/,
    '$1<i class="fas fa-clock meta-icon"></i>$2'
  );

  // Move meta below the title
  html = html.replace(
    /(<div class="article-meta[^"]*"[^>]*>[\s\S]*?<\/div><\/div>)(<h1 class="title[^"]*"[^>]*>[\s\S]*?<\/h1>)/,
    '$2$1'
  );

  return html;
});
