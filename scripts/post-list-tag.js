'use strict';

hexo.extend.tag.register('post_list', function(args) {
  const category = args[0];
  if (!category) return '';

  const posts = hexo.locals.get('posts').toArray()
    .filter(p => p.categories.toArray().some(c => c.name === category))
    .sort((a, b) => b.date - a.date);

  if (posts.length === 0) {
    return '<p class="post-list-empty">No posts yet.</p>';
  }

  return posts.map(p => {
    const date = p.date.format('MMM DD, YYYY');
    const desc = p.description ? `<p class="post-list-desc">${p.description}</p>` : '';
    const tags = p.tags.toArray()
      .map(t => `<span class="tag is-light">${t.name}</span>`)
      .join('');
    return `<div class="post-list-item">
  <div class="post-list-row">
    <a class="post-list-title" href="${p.path}">${p.title}</a>
    <span class="post-list-date">${date}</span>
  </div>
  ${desc}
  ${tags ? `<div class="post-list-tags">${tags}</div>` : ''}
</div>`;
  }).join('\n');
});
