// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/tree-d-3d/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'tree-d-3d',
	title: 'tree-d-3d',
	description: 'Barnes-Hut trees and fast multipole methods for charged particles in a plane.',
	pythonPackage: 'tree_d_3d',
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Tutorials', link: '/tutorials/' },
		{ label: 'Examples', link: '/examples/' },
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
			label: 'Tutorials',
			items: [{ autogenerate: { directory: 'tutorials' } }],
		},
		{
			label: 'Examples',
			items: [{ autogenerate: { directory: 'examples' } }],
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
