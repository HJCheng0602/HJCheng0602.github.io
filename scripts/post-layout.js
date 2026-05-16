// On post pages: hide right sidebar and expand main column to fill the space.
// Home/archives/tag pages keep the 3-column layout unchanged.
hexo.extend.filter.register('after_render:html', function(html, data) {
  const isPost = data && data.path && /^blog\/[^/]+\/index\.html$/.test(data.path);
  if (!isPost) return html;

  const style = `<style>
/* Post pages: hide right column, expand main to use freed space */
@media screen and (min-width: 1280px) {
  .column-right { display: none !important; }
  .column-main {
    flex: 1 1 0 !important;
    width: auto !important;
    max-width: none !important;
  }
}
/* Post pages: show TOC, hide categories */
.widget[data-type="toc"] { display: block !important; }
.widget[data-type="categories"] { display: none !important; }
</style>`;
  return html.replace('</head>', style + '</head>');
});
