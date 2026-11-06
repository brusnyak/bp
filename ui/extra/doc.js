document.addEventListener('DOMContentLoaded', () => {
  const bar = document.getElementById('bar');
  const onScroll = () => bar && bar.classList.toggle('scrolled', window.scrollY > 8);
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  // Table of contents from the page's section headings, with scroll spy.
  const list = document.getElementById('toc');
  if (!list) return;
  const heads = [...document.querySelectorAll('.prose section[id] > h2')];
  heads.forEach((h) => {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = '#' + h.parentElement.id;
    a.textContent = h.textContent;
    li.appendChild(a);
    list.appendChild(li);
  });
  const links = [...list.querySelectorAll('a')];
  if (!('IntersectionObserver' in window)) return;
  const spy = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      links.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id));
    });
  }, { rootMargin: '-20% 0px -65% 0px' });
  heads.forEach((h) => spy.observe(h.parentElement));
});
