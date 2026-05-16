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
    width: 82% !important;
    flex: none !important;
  }
}
</style>`;
  return html.replace('</head>', style + '</head>');
});
