# Releasing to npm

Pushing to `main` never publishes anything. A release is one workflow you start by hand: **Actions → Release to npm →
Run workflow**. It checks, builds, publishes, tags and pushes for you.

The first version, `0.1.0-alpha.0`, was published by hand on 2026-11-15 (see "The first release" below). Every
release after it goes through the workflow.

## Making a release

1. **Write the notes and push.** Add a `## 0.1.0` section (your version) to [CHANGELOG.md](../../CHANGELOG.md): it
   becomes the GitHub Release's text. Push to `main` and wait for CI to pass; the release refuses a commit whose CI
   run failed or is still running.
2. **Open Actions → Release to npm → Run workflow.** Leave the branch on `main` and fill in:
   - **Version**: the new version, without a `v`, for example `0.1.0-alpha.1`.
   - **npm dist-tag**: `alpha`, `beta`, `rc` or `next` for a prerelease, `latest` for a normal release.
   - **Dry run**: tick it to see what would be published without publishing anything.
3. **Run it.** It takes about five minutes. When it is green:
   - the version is on npm (`npm view @pmndrs/bvh dist-tags`);
   - the tag `v0.1.0-alpha.1` and its GitHub Release are on GitHub;
   - `main` has a new commit, "Release 0.1.0-alpha.1", so run `git pull` before your next push;
   - the docs site is rebuilding with the new version label.

Picking the version and tag:

| Release | Version | Tag | People install it with |
| --- | --- | --- | --- |
| Next alpha | `0.1.0-alpha.1` | `alpha` | `npm i @pmndrs/bvh@alpha` |
| Release candidate | `0.1.0-rc.0` | `rc` | `npm i @pmndrs/bvh@rc` |
| Release | `0.1.0` | `latest` | `npm i @pmndrs/bvh` |

A version can be published only once, ever. If a release was wrong, publish the next number.

`latest` points at `0.1.0-alpha.0` for now, because a package always has a `latest` and that is the only real
version. Prereleases go to their own tag, so `latest` stays there until `0.1.0`.

## One-time setup

Done once, on npmjs.com, by a maintainer of `@pmndrs/bvh`. The trusted publisher was added on 2026-11-15; if no
release runs through it by 2026-11-17, npm drops it and it has to be added again (step 1).

1. **Trusted publisher.** **@pmndrs/bvh → Settings → Trusted publishing → Add trusted publisher → GitHub Actions**,
   filled in exactly:
   - Organization or user: `pmndrs`
   - Repository: `bvh`
   - Workflow filename: `release.yml` (the filename only; the file is `.github/workflows/release.yml`)
   - Environment name: `npm`
   - Allowed actions: tick **`npm publish`** too. New trusted publishers only allow `npm stage publish`, and the
     workflow runs `npm publish`, so without it the Publish step fails.

   npm drops a trusted publisher that publishes nothing within two days. If yours is gone when you next release, add
   it again just before running the workflow.
2. **Block tokens.** **@pmndrs/bvh → Settings → Publishing access → Require two-factor authentication and disallow
   tokens → Update Package Settings.** The workflow keeps working, because trusted publishing does not use tokens,
   and nobody can publish with a leaked token. No `NPM_TOKEN` secret is needed anywhere.

Optional, on GitHub: **Settings → Environments → npm → Required reviewers** → yourself, so each run waits for your
click before it publishes.

## When something goes wrong

The run stops with a red error line saying why:

- **"CI run … is still in_progress"**: wait for CI on that commit, then run the release again.
- **"CI run … ended 'failure'"**: fix it on `main`, push, wait for CI, release from the new commit.
- **"… is already on npm"** or **"tag … already exists"**: that version is used up; pick the next one.
- **"… is a prerelease; pick next, alpha, beta or rc"**: a version with `-alpha` and the like can't go on `latest`.
- **E401, E403, ENEEDAUTH or "not allowed" at Publish**: the trusted publisher is missing (expired after two days),
  a field doesn't match exactly (workflow filename `release.yml`, environment `npm`), or `npm publish` isn't ticked
  under Allowed actions.
- **"published and tagged, but the release commit did not reach main"**: the package is out and tagged; only the
  version bump is missing from `main`. Run `git pull`, `git merge v<version>`, `git push`.
- **A release that should not be used**: `npm deprecate @pmndrs/bvh@<version> "<why>"` and publish a fixed version.
  npm only allows an unpublish within 72 hours, and the version number stays used up anyway.

## Publishing by hand

Only if the workflow can't be used. Use npm 11.15 or newer (`npm i -g npm@11`; older npm can't log in reliably or
approve staged versions), on a clean `main` whose CI passed. **Run the lines one at a time**: `npm login` and
`npm publish` stop to ask for input, and pasted lines that follow become their answers.

```powershell
git pull
pnpm install --frozen-lockfile
npm pkg set version=0.1.0-alpha.1
git add package.json
git commit -m "Release 0.1.0-alpha.1"
git tag -a v0.1.0-alpha.1 -m "@pmndrs/bvh 0.1.0-alpha.1"
npm login
npm publish --access public --tag alpha
git push origin main v0.1.0-alpha.1
```

`npm publish` runs lint, typecheck, test and build first (`prepublishOnly`, about three minutes), then asks for your
2FA code. A version published by hand has no provenance, since npm can only sign that inside CI.

npm may put the version in staging instead of publishing it. Then only a placeholder is public until you approve
it: npmjs.com → **Staged Packages** → **Approve** (2FA), or `npm stage list @pmndrs/bvh` and
`npm stage approve <id>`. Check the result with `npm dist-tag ls @pmndrs/bvh`.

## The first release

`0.1.0-alpha.0` went out by hand on 2026-11-15, because npm's trusted publisher settings live on the package's page
and the package didn't exist yet. The release commit is `9712461`, tagged `v0.1.0-alpha.0`. npm staged it: a
placeholder `0.0.0-stage` appeared first and stays in the version list, and after approval `alpha` and `latest` both
pointed at `0.1.0-alpha.0`. It has no provenance; workflow releases will.

## What the workflow does

[`.github/workflows/release.yml`](../../.github/workflows/release.yml), in order:

1. Checks the version: well formed, not on npm, not tagged, and not a prerelease on `latest`.
2. Checks that CI passed on the commit, which covers the GPU pages and the tinybvh WASM rebuild.
3. Runs lint, typecheck, test and build again, because this job builds what gets published.
4. Commits the version to `package.json` and tags it.
5. Publishes with provenance (a signed record on npm of which commit and workflow built the package), signing in
   through trusted publishing.
6. Pushes the tag and the commit to `main`, makes the GitHub Release from the version's section of `CHANGELOG.md`
   (GitHub's generated notes if there is none; marked as a prerelease for `-alpha` and the like), and starts the
   docs workflow.

A dry run runs `npm publish --dry-run` and prints the release notes: nothing is published or released, and the commit
and tag stay on the runner.
