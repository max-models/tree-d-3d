// Mount interactive Plotly figures from pre-exported figure JSON (`<PlotlyChart src=...>`).
//
// Ported from maxpic/docs (itself from struphy-hub.github.io): plotly.js comes from the npm
// package, loaded once, lazily, when the first chart scrolls into view; the layout is themed
// from the site's colours and re-themed when the reader switches between light and dark. A
// chart inside a closed <details> is drawn when it is opened.

type Layout = Record<string, any>;
interface FigureElement extends HTMLElement {
	_sourceLayout?: Layout;
	_hasFrames?: boolean;
}

const CONFIG = {
	responsive: true,
	displaylogo: false,
	modeBarButtonsToRemove: ['lasso2d', 'select2d'],
};

const mounted = new Set<FigureElement>();
let intersectionObserver: IntersectionObserver | undefined;
let themeObserver: MutationObserver | undefined;
let plotlyPromise: Promise<any> | undefined;

function getPlotly(): Promise<any> {
	// Elements with an id are globals: a heading "Exports" makes `window.exports` a DOM element,
	// which would confuse the module wrapper of the Plotly bundle.
	for (const name of ['exports', 'module', 'define']) {
		if ((window as any)[name] instanceof Element) (window as any)[name] = undefined;
	}
	plotlyPromise ??= import('plotly.js-dist-min').then((module: any) => module.default ?? module);
	return plotlyPromise;
}

function cssColor(name: string): string {
	return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function colors() {
	return {
		text: cssColor('--sl-color-white'),
		muted: cssColor('--sl-color-gray-3'),
		grid: cssColor('--sl-color-hairline-light'),
		background: cssColor('--sl-color-bg'),
		accent: cssColor('--sl-color-text-accent'),
	};
}

function fromMatplotlib(layout: Layout): boolean {
	return layout.meta?.converted_from === 'matplotlib';
}

function axisTheme(axis: Layout = {}, c: ReturnType<typeof colors>, light: boolean): Layout {
	// Axes drawn on a white Matplotlib plot area keep dark ink.
	const ink = light ? '#10222d' : c.text;
	const rule = light ? '#e4e0d4' : c.grid;
	return { ...axis, color: ink, gridcolor: rule, linecolor: light ? '#10222d' : c.grid, zerolinecolor: rule };
}

function themedUpdatemenus(updatemenus: Layout[] = [], c: ReturnType<typeof colors>): Layout[] {
	return updatemenus.map((menu) => ({
		...menu,
		bgcolor: c.background,
		bordercolor: c.accent,
		borderwidth: 1,
		font: { ...menu.font, color: c.text },
	}));
}

function themedSliders(sliders: Layout[] = [], c: ReturnType<typeof colors>): Layout[] {
	return sliders.map((slider) => ({
		...slider,
		bgcolor: c.accent,
		activebgcolor: c.text,
		bordercolor: c.grid,
		borderwidth: 1,
		font: { ...slider.font, color: c.text },
		currentvalue: { ...slider.currentvalue, font: { ...slider.currentvalue?.font, color: c.accent } },
		tickcolor: c.text,
	}));
}

// Plotly's default legend sits right of the plot area, on the page background.
function legendOutside(legend: Layout = {}): boolean {
	const x = legend.x ?? 1.02;
	const y = legend.y ?? 1;
	return x > 1 || x < 0 || y > 1 || y < 0;
}

export function themedLayout(layout: Layout = {}): Layout {
	const c = colors();
	const whitePlotArea = fromMatplotlib(layout);
	const whiteLegend = whitePlotArea && !legendOutside(layout.legend);
	const next: Layout = {
		...layout,
		// an exported figure may carry a fixed size (from writing a PNG): fill the container instead
		width: undefined,
		height: undefined,
		autosize: true,
		paper_bgcolor: 'rgba(0,0,0,0)',
		plot_bgcolor: whitePlotArea ? 'white' : 'rgba(0,0,0,0)',
		font: { ...layout.font, color: c.text },
		legend: {
			...layout.legend,
			bgcolor: whiteLegend ? 'rgba(255,255,255,0.8)' : 'rgba(0,0,0,0)',
			bordercolor: whiteLegend ? c.grid : 'rgba(0,0,0,0)',
			font: { ...layout.legend?.font, color: whiteLegend ? '#10222d' : c.text },
		},
		hoverlabel: { ...layout.hoverlabel, bgcolor: c.background, bordercolor: c.grid, font: { color: c.text } },
		updatemenus: themedUpdatemenus(layout.updatemenus, c),
		sliders: themedSliders(layout.sliders, c),
	};
	const axisNames = new Set(['xaxis', 'yaxis']);
	for (const name of Object.keys(layout)) if (/^[xyz]axis\d*$/.test(name)) axisNames.add(name);
	for (const name of axisNames) next[name] = axisTheme(layout[name], c, false);
	for (const name of Object.keys(layout).filter((key) => /^scene\d*$/.test(key))) {
		next[name] = {
			...layout[name],
			bgcolor: 'rgba(0,0,0,0)',
			xaxis: axisTheme(layout[name].xaxis, c, false),
			yaxis: axisTheme(layout[name].yaxis, c, false),
			zaxis: axisTheme(layout[name].zaxis, c, false),
		};
	}
	for (const name of Object.keys(layout).filter((key) => /^polar\d*$/.test(key))) {
		next[name] = {
			...layout[name],
			bgcolor: 'rgba(0,0,0,0)',
			angularaxis: axisTheme(layout[name].angularaxis, c, false),
			radialaxis: axisTheme(layout[name].radialaxis, c, false),
		};
	}
	return next;
}

function setupPlayButton(root: FigureElement, Plotly: any): void {
	let button = root.querySelector<HTMLButtonElement>('[data-plot-play]');
	let replay = root.querySelector<HTMLButtonElement>('[data-plot-replay]');
	if (!button || !replay) return;
	const label = button.querySelector('[data-plot-play-label]');
	button.hidden = false;
	replay.hidden = false;
	replay.addEventListener('click', () => {
		const enabled = replay!.getAttribute('aria-pressed') !== 'true';
		replay!.setAttribute('aria-pressed', String(enabled));
	});
	const setIdle = () => {
		button!.dataset.playing = 'false';
		if (label) label.textContent = 'Play';
	};
	const play = (fromCurrent: boolean) => {
		Plotly.animate(root, null, {
			frame: { duration: 60, redraw: true },
			transition: { duration: 0 },
			fromcurrent: fromCurrent,
		}).then(() => {
			if (button!.dataset.playing !== 'true') return;
			if (replay!.getAttribute('aria-pressed') === 'true') play(false);
			else setIdle();
		});
	};
	button.addEventListener('click', () => {
		if (button!.dataset.playing === 'true') {
			setIdle();
			Plotly.animate(root, [null], { frame: { duration: 0, redraw: false }, mode: 'immediate' });
			return;
		}
		button!.dataset.playing = 'true';
		if (label) label.textContent = 'Pause';
		play(true);
	});
}

async function render(root: FigureElement): Promise<void> {
	if (root.dataset.plotMounted) return;
	root.dataset.plotMounted = 'true';
	try {
		const [Plotly, response] = await Promise.all([getPlotly(), fetch(root.dataset.plotSrc!)]);
		if (!response.ok) throw new Error(`HTTP ${response.status}`);
		const figure = await response.json();
		root._sourceLayout = figure.layout ?? {};
		root._hasFrames = Boolean(figure.frames?.length);
		root.querySelector('[data-plot-status]')?.remove();
		const layout = themedLayout(root._sourceLayout);
		await Plotly.newPlot(root, figure.data ?? [], layout, CONFIG);
		if (root._hasFrames) {
			await Plotly.addFrames(root, figure.frames);
			// the figure's own slider and Play buttons (plasma-plots animations) stay; the
			// round buttons below are for figures that bring frames without controls
			if (!layout.updatemenus?.length) setupPlayButton(root, Plotly);
		}
		root.setAttribute('aria-busy', 'false');
		mounted.add(root);
	} catch (error) {
		console.error(`Unable to load Plotly figure ${root.dataset.plotSrc}:`, error);
		root.textContent = 'Interactive figure unavailable.';
		root.classList.add('plot-error');
		root.setAttribute('aria-busy', 'false');
		root.dataset.plotMounted = 'error';
	}
}

function ensureObservers(): void {
	intersectionObserver ??= new IntersectionObserver(
		(entries) => {
			for (const entry of entries) {
				if (!entry.isIntersecting) continue;
				intersectionObserver!.unobserve(entry.target);
				void render(entry.target as FigureElement);
			}
		},
		{ rootMargin: '500px 0px' }
	);
	if (!themeObserver) {
		themeObserver = new MutationObserver(async () => {
			if (!mounted.size) return;
			const Plotly = await getPlotly();
			for (const root of mounted) {
				if (!document.contains(root)) {
					mounted.delete(root);
					continue;
				}
				Plotly.relayout(root, themedLayout(root._sourceLayout));
			}
		});
		themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
	}
}

export function mountPlotlyFigures(): void {
	if (typeof document === 'undefined') return;
	ensureObservers();
	for (const root of document.querySelectorAll<FigureElement>('[data-plotly-figure]:not([data-plot-mounted])')) {
		// a chart in a closed <details> would be laid out at zero width: draw it when opened
		const folded = root.closest('details');
		if (folded && !folded.open) {
			folded.addEventListener('toggle', () => folded.open && void render(root));
		} else if (root.dataset.eager === 'true') {
			void render(root);
		} else {
			intersectionObserver!.observe(root);
		}
	}
}
