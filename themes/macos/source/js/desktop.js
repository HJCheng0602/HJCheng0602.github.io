/* macOS desktop theme — interactions */
(function () {
  'use strict';

  /* ── Clocks ────────────────────────────────────────────────── */
  var DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  function pad(n) { return n < 10 ? '0' + n : '' + n; }

  function hour12(d) {
    var h = d.getHours() % 12;
    return h === 0 ? 12 : h;
  }

  function updateClocks() {
    var d = new Date();
    var menuClock = document.getElementById('menu-clock');
    if (menuClock) {
      var ampm = d.getHours() < 12 ? 'AM' : 'PM';
      menuClock.textContent = DAYS[d.getDay()] + ' ' + MONTHS[d.getMonth()] + ' ' +
        d.getDate() + '  ' + hour12(d) + ':' + pad(d.getMinutes()) + ' ' + ampm;
    }
    var lockTime = document.getElementById('lock-time');
    if (lockTime) lockTime.textContent = hour12(d) + ':' + pad(d.getMinutes());
    var lockDate = document.getElementById('lock-date');
    if (lockDate) {
      lockDate.textContent = DAYS[d.getDay()] + ', ' + MONTHS[d.getMonth()] + ' ' + d.getDate();
    }
  }
  updateClocks();
  setInterval(updateClocks, 5000);

  /* ── Lock screen ───────────────────────────────────────────── */
  var lock = document.getElementById('lockscreen');
  if (lock) {
    var alreadyUnlocked = false;
    try { alreadyUnlocked = sessionStorage.getItem('mac_unlocked') === '1'; } catch (e) { /* private mode */ }
    if (alreadyUnlocked) {
      lock.classList.add('unlocked');
      document.body.classList.remove('is-locked');
    }
    lock.addEventListener('click', function () {
      lock.classList.add('unlocked');
      document.body.classList.remove('is-locked');
      try { sessionStorage.setItem('mac_unlocked', '1'); } catch (e) {}
    });
  }

  /* ── Window manager ────────────────────────────────────────── */
  var zTop = 10;

  function bringToFront(win) { win.style.zIndex = ++zTop; }

  document.querySelectorAll('.window').forEach(function (win) {
    win.addEventListener('pointerdown', function () { bringToFront(win); });

    var bar = win.querySelector('.window-titlebar');
    if (!bar) return;
    var startX = 0, startY = 0, baseLeft = 0, baseTop = 0, minTop = 0, dragging = false;

    bar.addEventListener('pointerdown', function (e) {
      if (e.target.closest('.tl') || win.classList.contains('maximized')) return;
      // Windows are forced fullscreen on small screens; dragging is disabled.
      if (window.matchMedia('(max-width: 720px)').matches) return;
      e.preventDefault();
      dragging = true;
      bringToFront(win);
      var rect = win.getBoundingClientRect();
      var parent = win.offsetParent || document.body;
      var prect = parent.getBoundingClientRect();
      // Normalize: switch from transform-centering to explicit left/top,
      // converting viewport coords into the offset-parent's coord system.
      win.style.transform = 'none';
      baseLeft = rect.left - prect.left;
      baseTop = rect.top - prect.top;
      win.style.left = baseLeft + 'px';
      win.style.top = baseTop + 'px';
      // Keep the titlebar below the menubar (menubar bottom = viewport y 30).
      minTop = 30 - prect.top;
      startX = e.clientX; startY = e.clientY;
    });
    // Listen on document so the drag never gets stuck when the pointer
    // leaves the titlebar or a pointerup is missed.
    document.addEventListener('pointermove', function (e) {
      if (!dragging) return;
      if (e.buttons === 0) { dragging = false; return; } // missed pointerup
      win.style.left = baseLeft + (e.clientX - startX) + 'px';
      win.style.top = Math.max(minTop, baseTop + (e.clientY - startY)) + 'px';
    });
    document.addEventListener('pointerup', function () { dragging = false; });
    document.addEventListener('pointercancel', function () { dragging = false; });
  });

  document.addEventListener('click', function (e) {
    var btn = e.target.closest('[data-win]');
    if (!btn) return;
    var win = btn.closest('.window');
    if (!win) return;
    var action = btn.getAttribute('data-win');
    if (action === 'close' || action === 'min') {
      var url = win.getAttribute('data-close-url');
      var back = win.getAttribute('data-close-back') === '1';
      if (url || back) {
        // Page-level window: close & minimize both leave the page.
        // Minimize plays the shrink animation first, then navigates.
        var go = function () {
          if (url) window.location.href = url;
          else if (window.history.length > 1) window.history.back();
          else window.location.href = '/';
        };
        if (action === 'min') {
          win.classList.add('minimized');
          setTimeout(go, 190);
        } else {
          go();
        }
        return;
      }
      // Desktop window: close & minimize both hide it (shrink animation).
      win.classList.add('minimized');
    } else if (action === 'max') {
      win.classList.toggle('maximized');
    }
  });

  /* ── Desktop icons: double-click opens a Finder window ─────── */
  /* Touch devices (coarse pointer) use a single tap instead.     */
  var openEvt = window.matchMedia('(pointer: coarse)').matches ? 'click' : 'dblclick';
  document.querySelectorAll('[data-open-window]').forEach(function (el) {
    el.addEventListener(openEvt, function () {
      var win = document.getElementById(el.getAttribute('data-open-window'));
      if (!win) return;
      win.hidden = false;
      win.classList.remove('minimized');
      bringToFront(win);
    });
  });

  /* ── Desktop icons: double-click navigates to a URL ────────── */
  document.querySelectorAll('[data-open-url]').forEach(function (el) {
    el.addEventListener(openEvt, function () {
      window.location.href = el.getAttribute('data-open-url');
    });
  });

  /* ── Widget chips: single-click opens a window ─────────────── */
  document.querySelectorAll('[data-open-win]').forEach(function (el) {
    el.addEventListener('click', function () {
      var win = document.getElementById(el.getAttribute('data-open-win'));
      if (!win) return;
      win.hidden = false;
      win.classList.remove('minimized');
      bringToFront(win);
    });
  });

  /* ── Dock special actions ──────────────────────────────────── */
  document.querySelectorAll('.dock-item').forEach(function (item) {
    var action = item.getAttribute('data-action');
    if (action === 'all-posts') {
      item.addEventListener('click', function (e) {
        var win = document.getElementById('finder-all-posts');
        if (win) {
          e.preventDefault();
          win.hidden = false;
          win.classList.remove('minimized');
          bringToFront(win);
        }
        // otherwise fall through to /archives
      });
    } else if (action === 'trash') {
      item.addEventListener('click', function (e) {
        e.preventDefault();
        var icon = item.querySelector('.dock-icon');
        icon.style.animation = 'none';
        void icon.offsetWidth;
        icon.style.animation = 'dock-shake .4s';
      });
    }
  });

  /* ── Spotlight search ──────────────────────────────────────── */
  var spotlight = document.getElementById('spotlight');
  var spInput = document.getElementById('spotlight-input');
  var spResults = document.getElementById('spotlight-results');
  var spBtn = document.getElementById('spotlight-btn');
  var spIndex = null;

  function loadIndex() {
    if (spIndex) return Promise.resolve(spIndex);
    return fetch('/search.xml')
      .then(function (r) { return r.text(); })
      .then(function (text) {
        var doc = new DOMParser().parseFromString(text, 'text/xml');
        spIndex = Array.prototype.map.call(doc.querySelectorAll('entry'), function (en) {
          var t = en.querySelector('title');
          var u = en.querySelector('url');
          var url = u ? u.textContent : '';
          url = url.replace(/^\/\//, '/');
          return { title: t ? t.textContent : '', url: url };
        });
        return spIndex;
      })
      .catch(function () { spIndex = []; return spIndex; });
  }

  function renderResults(q) {
    var query = q.trim().toLowerCase();
    if (!query) { spResults.innerHTML = ''; return; }
    var hits = spIndex.filter(function (en) {
      return en.title.toLowerCase().indexOf(query) !== -1;
    }).slice(0, 8);
    if (!hits.length) {
      spResults.innerHTML = '<li><span class="sp-empty">No Results</span></li>';
      return;
    }
    spResults.innerHTML = hits.map(function (h, i) {
      return '<li><a href="' + h.url + '"' + (i === 0 ? ' class="active"' : '') + '>' +
        '<span>' + h.title.replace(/</g, '&lt;') + '</span></a></li>';
    }).join('');
  }

  function openSpotlight() {
    spotlight.hidden = false;
    spInput.value = '';
    spResults.innerHTML = '';
    spInput.focus();
    loadIndex();
  }
  function closeSpotlight() { spotlight.hidden = true; }

  if (spBtn && spotlight) {
    spBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      if (spotlight.hidden) openSpotlight(); else closeSpotlight();
    });
    spotlight.addEventListener('click', function (e) {
      if (e.target === spotlight) closeSpotlight();
    });
    spInput.addEventListener('input', function () {
      loadIndex().then(function () { renderResults(spInput.value); });
    });
    spInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        var first = spResults.querySelector('a');
        if (first) window.location.href = first.getAttribute('href');
      } else if (e.key === 'Escape') {
        closeSpotlight();
      }
    });
    document.addEventListener('keydown', function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        if (spotlight.hidden) openSpotlight(); else closeSpotlight();
      }
    });
  }

  /* ── Widget scrollbars: reveal only while scrolling ────────── */
  document.querySelectorAll('.widget-body').forEach(function (body) {
    var timer = null;
    body.addEventListener('scroll', function () {
      body.classList.add('scrolling');
      clearTimeout(timer);
      timer = setTimeout(function () { body.classList.remove('scrolling'); }, 700);
    });
  });

  /* ── TOC scroll-spy: highlight the heading being read ──────── */
  (function initScrollSpy() {
    var main = document.querySelector('.article-main');
    var sidebar = document.querySelector('.article-toc-sidebar');
    if (!main || !sidebar) return;
    var links = Array.prototype.slice.call(sidebar.querySelectorAll('a[href^="#"]'));
    if (!links.length) return;
    var items = [];
    links.forEach(function (link) {
      var id = decodeURIComponent(link.getAttribute('href').slice(1));
      var el = document.getElementById(id);
      if (el) items.push({ el: el, link: link });
    });
    if (!items.length) return;
    var ticking = false;
    function update() {
      ticking = false;
      var mainTop = main.getBoundingClientRect().top;
      var current = items[0];
      for (var i = 0; i < items.length; i++) {
        if (items[i].el.getBoundingClientRect().top - mainTop <= 90) current = items[i];
        else break;
      }
      links.forEach(function (l) { l.classList.remove('active'); });
      if (current) current.link.classList.add('active');
    }
    main.addEventListener('scroll', function () {
      if (!ticking) { ticking = true; requestAnimationFrame(update); }
    });
    update();
  })();

  /* ── Daily wallpaper rotation ──────────────────────────────── */
  (function initWallpaper() {
    var wp = document.querySelector('.wallpaper');
    if (!wp) return;
    var list = [];
    try { list = JSON.parse(wp.getAttribute('data-wallpapers') || '[]'); } catch (e) { list = []; }
    if (!list.length) return;
    var idx = Math.floor(Date.now() / 86400000) % list.length;
    var url = '/img/wallpapers/' + list[idx] + '.jpg';
    var img = new Image();
    img.onload = function () {
      wp.style.background =
        "linear-gradient(rgba(8,14,26,.34), rgba(8,14,26,.34)), url('" + url + "') center/cover no-repeat, " +
        "linear-gradient(160deg, #0d1b2e 0%, #10263f 45%, #0a1a2c 100%)";
    };
    img.src = url;
  })();

  /* ── Menubar dropdowns & panels ────────────────────────────── */
  (function initMenus() {
    var roots = Array.prototype.slice.call(document.querySelectorAll('.menu-root'));
    if (!roots.length) return;

    function panelOf(root) { return root.querySelector('.menu-dropdown, .menu-panel'); }
    function closeAll(except) {
      roots.forEach(function (r) {
        if (r === except) return;
        r.classList.remove('open');
        var p = panelOf(r);
        if (p) p.hidden = true;
      });
    }

    roots.forEach(function (root) {
      var trigger = root.querySelector('.menu-trigger');
      var panel = panelOf(root);
      if (!trigger || !panel) return;
      trigger.addEventListener('click', function (e) {
        e.stopPropagation();
        var willOpen = !root.classList.contains('open');
        closeAll(root);
        root.classList.toggle('open', willOpen);
        panel.hidden = !willOpen;
        if (willOpen && root.getAttribute('data-menu') === 'clock') renderCalendar();
      });
    });
    document.addEventListener('click', function () { closeAll(null); });

    document.querySelectorAll('.menu-item').forEach(function (item) {
      item.addEventListener('click', function () {
        var action = item.getAttribute('data-action');
        var url = item.getAttribute('data-url');
        closeAll(null);
        runAction(action, url);
      });
    });

    function runAction(action, url) {
      if (action === 'nav' && url) { window.location.href = url; return; }
      if (action === 'lock') {
        var ls = document.getElementById('lockscreen');
        if (ls) {
          ls.classList.remove('unlocked');
          document.body.classList.add('is-locked');
          try { sessionStorage.removeItem('mac_unlocked'); } catch (e) {}
        }
        return;
      }
      if (action === 'fullscreen') {
        if (!document.fullscreenElement && document.documentElement.requestFullscreen) document.documentElement.requestFullscreen();
        else if (document.exitFullscreen) document.exitFullscreen();
        return;
      }
      if (action === 'copy') { document.execCommand('copy'); return; }
      if (action === 'selectall') { document.execCommand('selectAll'); return; }
      if (action === 'welcome') {
        var w = document.getElementById('welcome-window');
        if (w) { w.hidden = false; w.classList.remove('minimized'); bringToFront(w); }
        return;
      }
      if (action === 'close' || action === 'min' || action === 'zoom') {
        var win = topWindow();
        if (!win) return;
        if (action === 'zoom') { win.classList.toggle('maximized'); return; }
        var curl = win.getAttribute('data-close-url');
        var back = win.getAttribute('data-close-back') === '1';
        if (curl || back) {
          var go = function () {
            if (curl) window.location.href = curl;
            else if (window.history.length > 1) window.history.back();
            else window.location.href = '/';
          };
          if (action === 'min') { win.classList.add('minimized'); setTimeout(go, 190); }
          else go();
          return;
        }
        win.classList.add('minimized');
      }
      // noop: intentional no-op
    }

    function topWindow() {
      var wins = Array.prototype.slice.call(document.querySelectorAll('.window')).filter(function (w) {
        return !w.hidden && !w.classList.contains('minimized');
      });
      if (!wins.length) return null;
      return wins.reduce(function (a, b) {
        return (parseInt(a.style.zIndex || 0, 10) >= parseInt(b.style.zIndex || 0, 10)) ? a : b;
      });
    }

    function renderCalendar() {
      var grid = document.getElementById('cal-grid');
      var title = document.getElementById('cal-title');
      if (!grid || !title) return;
      var now = new Date();
      var y = now.getFullYear(), mo = now.getMonth();
      title.textContent = y + ' / ' + (mo + 1);
      var first = new Date(y, mo, 1).getDay();
      var days = new Date(y, mo + 1, 0).getDate();
      var dows = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];
      var html = dows.map(function (d) { return '<div class="cal-dow">' + d + '</div>'; }).join('');
      for (var i = 0; i < first; i++) html += '<div class="cal-day empty"></div>';
      for (var d = 1; d <= days; d++) {
        html += '<div class="cal-day' + (d === now.getDate() ? ' today' : '') + '">' + d + '</div>';
      }
      grid.innerHTML = html;
    }
  })();
})();
