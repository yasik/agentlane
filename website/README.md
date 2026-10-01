# AgentLane website

Build and run the AgentLane website with Node.js 24 and pnpm 10.29.3.
Run the commands in this page from the `website/` directory.
Keep the full repository checkout available: the documentation portal reads
Markdown directly from the root `docs/` directory.

## Run locally

Install dependencies and start the development server:

```sh
make install
make dev
```

Open the local URL printed in the terminal.

## Build and preview

Build the website and preview the static output:

```sh
make build
make preview
```

The build writes the website and `/docs/` portal to `dist/`. It excludes
`docs/plans/`. Edit documentation in root `docs/`, then rebuild the website.

## Check the build

Check formatting, types, generated pages, and documentation links:

```sh
make check
```

For a Vercel project with `website` as its root directory, enable **Include
source files outside of the Root Directory in the Build Step**. The build
needs root `docs/` and the repository files that its links reference. Make
sure that the project's **Ignored Build Step** does not skip `docs/` changes.
For more information, see [Vercel's shared source file setup](https://vercel.com/docs/monorepos/monorepo-faq#can-i-share-source-files-between-projects-are-shared-packages-supported).
