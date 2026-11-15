# Releasing to npm

Pushing to `main` never publishes anything. A release is one workflow you start by hand: **Actions → Release to npm →
Run workflow**. It checks, builds, publishes, tags and pushes for you.

## Making a release

1. **Push to `main` and wait for CI to pass.** The release refuses a commit whose CI run failed or is still running.
2. **Open Actions → Release to npm → Run workflow.** Leave the branch on `main` and fill in:
   - **Version**: the new version, without a `v`, for example `0.1.0-alpha.0`.
   - **npm dist-tag**: `alpha`, `beta`, `rc` or `next` for a prerelease, `latest` for a normal release.
   - **Dry run**: tick it to see what would be published without publishing anything.
3. **Run it.** It takes about five minutes. When it is green:
   - the version is on npm (`npm view @pmndrs/bvh versions`);
   - the tag `v0.1.0-alpha.0` is on GitHub;
   - `main` has a new commit, "Release 0.1.0-alpha.0", so run `git pull` before your next push;
   - the docs site is rebuilding with the new version label.

Picking the version and tag:

| Release | Version | Tag | People install it with |
| --- | --- | --- | --- |
| First alpha | `0.1.0-alpha.0` | `alpha` | `npm i @pmndrs/bvh@alpha` |
| Next alpha | `0.1.0-alpha.1` | `alpha` | `npm i @pmndrs/bvh@alpha` |
| Release candidate | `0.1.0-rc.0` | `rc` | `npm i @pmndrs/bvh@rc` |
| Release | `0.1.0` | `latest` | `npm i @pmndrs/bvh` |

A version can be published only once, ever. If a release was wrong, publish the next number.

## One-time setup

Do steps 1 to 3 before the first release, and steps 4 and 5 right after it.

1. **npm account.** You need an npmjs.com account with two-factor authentication on, and publish rights to the
   `@pmndrs` scope (an owner of the pmndrs npm organisation adds you to a team with write access).
2. **A token for the first publish.** Trusted publishing (step 4) is set up on the package's npm page, and that page
   only exists after the first publish. So the first release signs in with a token:
   - npmjs.com → your profile picture → **Access Tokens** → **Generate New Token**.
   - Token name: `bvh first release`.
   - Tick **Bypass two-factor authentication**, so the workflow can publish without a 2FA code.
   - **Packages and scopes** → Permissions: **Read and write (publish and stage)** → **Only select packages and
     scopes** → `@pmndrs`.
   - Expiration: the shortest option that covers when you will release (7 days is plenty).
   - **Generate Token** and copy it.

   npm plans to stop direct publishing with tokens in January 2027. Do the first release before then; after it,
   trusted publishing needs no token.
3. **Give the token to GitHub.** github.com/pmndrs/bvh → **Settings → Environments → npm** (the release workflow
   created this environment on its first run) → **Environment secrets → Add environment secret**. Name it
   `NPM_TOKEN` and paste the token. Only the release job can read it there.

   Optional, on the same page: **Required reviewers** → yourself, so every release waits for your click before it runs.

   Now make the first release (above).
4. **Switch to trusted publishing.** npmjs.com → **@pmndrs/bvh → Settings → Trusted publishing → Add trusted
   publisher → GitHub Actions**, and fill in exactly:
   - Organization or user: `pmndrs`
   - Repository: `bvh`
   - Workflow filename: `release.yml`
   - Environment name: `npm`

   npm drops this setting if nothing publishes through it within two days. If the next release is further off, add it
   when you are about to release.
5. **Remove the token.** Delete the `NPM_TOKEN` secret on GitHub, and delete the token on npmjs.com (Access Tokens).
   Then on npmjs.com, **@pmndrs/bvh → Settings → Publishing access → Require two-factor authentication and disallow
   tokens → Update Package Settings**. Releases keep working, because trusted publishing does not use tokens, and
   nobody can publish with a leaked token.

## When something goes wrong

The run stops with a red error line saying why:

- **"CI run … is still in_progress"**: wait for CI on that commit, then run the release again.
- **"CI run … ended 'failure'"**: fix it on `main`, push, wait for CI, release from the new commit.
- **"… is already on npm"** or **"tag … already exists"**: that version is used up; pick the next one.
- **"… is a prerelease; pick next, alpha, beta or rc"**: a version with `-alpha` and the like can't go on `latest`.
- **E401, E403 or ENEEDAUTH at Publish**: the token is missing or expired (first release), or the trusted publisher
  fields don't match exactly (later releases: check the workflow filename and the environment `npm`).
- **"published and tagged, but the release commit did not reach main"**: the package is out and tagged; only the
  version bump is missing from `main`. Run `git pull`, `git merge v<version>`, `git push`.
- **A release that should not be used**: `npm deprecate @pmndrs/bvh@<version> "<why>"` and publish a fixed version.
  npm only allows an unpublish within 72 hours, and the version number stays used up anyway.

## What the workflow does

[`.github/workflows/release.yml`](../../.github/workflows/release.yml), in order:

1. Checks the version: well formed, not on npm, not tagged, and not a prerelease on `latest`.
2. Checks that CI passed on the commit, which covers the GPU pages and the tinybvh WASM rebuild.
3. Runs lint, typecheck, test and build again, because this job builds what gets published.
4. Commits the version to `package.json` and tags it.
5. Publishes with provenance (a signed record on npm of which commit and workflow built the package). `main` keeps
   `"private": true`; the workflow drops it only for the publish, so a stray `npm publish` on a laptop is refused.
6. Pushes the tag and the commit to `main` and starts the docs workflow.

A dry run stops after step 5's `npm publish --dry-run`: nothing is published, committed, tagged or pushed.
