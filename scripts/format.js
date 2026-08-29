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
  return data;
});
