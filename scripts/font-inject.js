// Inject font <link> tags into every page's <head>.
// Using Hexo injector avoids the Icarus meta config's semicolon/equals
// parsing that truncates URLs containing query strings.
hexo.extend.injector.register('head_begin', () => `
<link rel="preconnect" href="https://fonts.loli.net" crossorigin>
<link rel="stylesheet" href="https://fonts.loli.net/css2?family=Noto+Serif+SC:wght@300;400;500;700&family=Noto+Sans+SC:wght@300;400;500&family=JetBrains+Mono:ital,wght@0,400;0,500;1,400&display=swap">
`);
