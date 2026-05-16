// Inject font + custom CSS <link> tags into every page's <head>.
// Must use Hexo injector — Icarus meta config generates <meta> tags (ignored
// by browsers for stylesheets) and truncates URLs with query strings.
hexo.extend.injector.register('head_begin', () => `
<link rel="preconnect" href="https://fonts.loli.net" crossorigin>
<link rel="stylesheet" href="https://fonts.loli.net/css2?family=Neuton:ital,wght@0,200;0,300;0,400;0,700;0,800;1,400&family=Oswald:wght@200..700&family=Noto+Serif+SC:wght@300;400;500;700&family=JetBrains+Mono:ital,wght@0,400;0,500;1,400&display=swap">
<link rel="stylesheet" href="/css/custom.css">
`);
