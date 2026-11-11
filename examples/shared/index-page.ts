// The index page's cards, from the one list of examples.
import { EXAMPLES, SOURCE_TREE } from './examples.ts';

const grid = document.getElementById('three')!;
for (const example of EXAMPLES) {
    const card = document.createElement('div');
    card.className = 'card';
    const title = document.createElement('a');
    title.className = 'title';
    title.href = `./${example.path}/`;
    title.textContent = example.path;
    const description = document.createElement('div');
    description.className = 'description';
    description.textContent = example.description;
    const source = document.createElement('a');
    source.className = 'source';
    source.href = `${SOURCE_TREE}${example.path}`;
    source.target = '_blank';
    source.rel = 'noopener';
    source.textContent = 'view source ↗';
    card.append(title, description, source);
    grid.append(card);
}
