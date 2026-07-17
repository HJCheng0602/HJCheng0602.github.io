'use strict';

// Group posts into desktop topic folders per theme config.
// Matching by post.slug; unmatched posts fall into "Misc".
hexo.extend.helper.register('topic_folders', function () {
  const folders = this.theme.topic_folders || [];
  const posts = this.site.posts.toArray()
    .sort((a, b) => b.date - a.date);
  const assigned = new Set();

  const result = folders.map(f => {
    let re = null;
    if (f.match) {
      try { re = new RegExp(f.match); } catch (e) { re = null; }
    }
    const matched = posts.filter(p => {
      if (assigned.has(p.slug)) return false;
      if (f.posts && f.posts.indexOf(p.slug) !== -1) return true;
      if (re && re.test(p.slug)) return true;
      return false;
    });
    matched.forEach(p => assigned.add(p.slug));
    return { name: f.name, posts: matched };
  });

  const rest = posts.filter(p => !assigned.has(p.slug));
  if (rest.length) result.push({ name: 'Misc', posts: rest });
  return result;
});

// Format date as MM-DD-YYYY (kept from the old article-meta style).
hexo.extend.helper.register('mac_date', function (date) {
  if (!date) return '';
  const d = date.toDate ? date.toDate() : new Date(date);
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  return mm + '-' + dd + '-' + d.getFullYear();
});

// Reading time text, e.g. "12 min read" (uses word-counter data if present).
hexo.extend.helper.register('mac_readtime', function (post) {
  if (post.min2read) {
    return (typeof post.min2read === 'string' ? post.min2read : post.min2read.text || '') + '';
  }
  return '';
});
