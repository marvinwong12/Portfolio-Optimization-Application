// Click a chart thumbnail to view it full size; click anywhere / Esc to close.
document.addEventListener('DOMContentLoaded', function () {
    const box = document.createElement('div');
    box.className = 'lightbox';
    box.hidden = true;
    const big = document.createElement('img');
    box.appendChild(big);
    document.body.appendChild(box);
    const close = () => { box.hidden = true; };
    box.addEventListener('click', close);
    document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
    document.querySelectorAll('img.chart').forEach(img => {
        img.addEventListener('click', () => { big.src = img.src; big.alt = img.alt; box.hidden = false; });
    });
    const print = document.getElementById('print-report');
    if (print) print.addEventListener('click', () => window.print());
});
