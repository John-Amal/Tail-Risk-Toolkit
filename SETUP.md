# Pushing this to GitHub

```bash
cd tailrisk-mcp
git init
git add .
git commit -m "EVT tail-risk MCP server: GPD fits, VaR/ES, return levels, VaR backtests"
git branch -M main
git remote add origin https://github.com/John-Amal/tailrisk-mcp.git
git push -u origin main
```

Create the empty repo on GitHub first (no README, no .gitignore, since both are here).

## Before you push

1. Run `pytest` yourself and watch it pass locally.
2. Run `examples/demo.py` and make sure the numbers make sense to you.
3. Wire it into Claude Desktop with the config in the README, then ask it something like
   "fit a GPD to this series and tell me whether the 99% VaR model passes backtesting"
   and watch the tool calls happen. That round trip is the thing worth being able to
   describe in an interview.
4. Read `evt.py` line by line. The maths is your home ground; the MCP wrapper is the
   only new part, and it is thin on purpose.

## Known things a reviewer might ask

- Why no GARCH filter? Answered in the README limitations section. McNeil and Frey (2000)
  is the standard reference if you want to add it, and it would be the strongest next commit.
- Why no confidence intervals on VaR? Also flagged as a limitation. A bootstrap over the
  exceedances is roughly 20 lines and would be a good second commit.
- Why is the calibration test averaged over seeds? Because the Christoffersen statistic is
  noisy with ~20 breaches; the test docstring explains this.
