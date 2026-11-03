// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/tree-d-3d/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'tree-d-3d',
	title: 'tree-d-3d',
	description: 'Tree Dynamics in 3D: Barnes-Hut trees and fast multipole methods for charged particles.',
	pythonPackage: 'tree_d_3d',
	// the method cards: a Numerics section with the overview (numerics/index.mdx) and one page per
	// module, from the trees to the solvers built on them, and a Numerics section on each method's
	// API page
	methodcard: {
		pages: true,
		index: false,
		labels: {
			'tree_d_3d.flat_tree': 'Trees and FMM',
			'tree_d_3d.reference': 'Reference tree',
			'tree_d_3d.bodies': 'Macroparticles',
			'tree_d_3d.simulation': 'Particle pusher',
			'tree_d_3d.strip': 'Grounded strip',
			'tree_d_3d.electrodes': 'Electrodes',
			'tree_d_3d.smooth': 'Smooth-membership FMM',
			'tree_d_3d.gravity': 'Gravity tree',
		},
	},
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Examples', link: '/examples/' },
		{ label: 'Numerics', link: '/numerics/' },
		{ label: 'Development', link: '/development/docs/' },
		{ label: 'API', link: '/api/tree_d_3d/' },
	],
	sidebar: [
		{
			label: 'Getting started',
			items: [
				{ label: 'Installation', slug: 'getting-started/installation' },
				{ label: 'Quickstart', slug: 'getting-started/quickstart' },
			],
		},
		{
			label: 'Examples',
			items: [
				{ autogenerate: { directory: 'examples' } },
				{ label: 'Scripts', items: [{ autogenerate: { directory: 'example-scripts' } }] },
			],
		},
		{
			label: 'Development',
			items: [
				{ label: 'Building the docs', slug: 'development/docs' },
				{ label: 'Publishing', slug: 'development/publishing' },
			],
		},
	],
});
