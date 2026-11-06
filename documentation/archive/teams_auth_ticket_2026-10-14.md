# Support ticket — Microsoft Teams sign-in loop on Mac (Authenticator)

Status: draft, ready to send to STU IT / Microsoft 365 helpdesk.
Fill the [BRACKETS], copy the message below into the helpdesk form or email.

## Send-to message (copy/paste)

Subject: Teams on macOS — cannot sign in, Authenticator activation loop

Hello,

I cannot sign in to Microsoft Teams on my Mac because of an
Authenticator activation loop:

- Teams asks me to approve/activate the account in the Microsoft
  Authenticator app.
- The Authenticator app itself asks for an approval code that is
  supposed to come from the app — so the two apps keep requesting
  the code from each other and I never get past activation.

Account: [STU LOGIN / UPN, e.g. xname@stuba.sk]
Affected device: Mac, macOS [VERSION], Teams build [VERSION, e.g. 24293.x],
  Authenticator on [iPhone/Android, app version].
What I already tried: [reinstall Teams / remove account and re-add /
  Authenticator backup-restore / date-time auto-sync — delete whichever
  does not apply].
Impact: I need Teams working for thesis-defense meetings; currently
  blocked on every sign-in attempt.

Please reset my MFA/Authenticator registration (or re-issue activation)
and tell me the exact re-enrollment steps.

Thank you,
[NAME], [FACULTY, STUDY PROGRAMME], tel. [PHONE]

## Notes for the owner

- The loop described ("app asks for a code from itself") is the classic
  orphaned MFA registration: fix is server-side (admin resets the
  authentication methods), not another reinstall.
- Until support replies: present over Meet/Zoom in a browser, or play the
  recorded fallback (`processed/e2e_ensk_sk.wav` + Voice Lab static page).
  Do not burn demo time on Teams login.
- After the reset: enroll Authenticator fresh (QR scan, not restore), keep
  one fallback method (SMS or HR PIN), test Teams login the day before
  any defense-related call.
