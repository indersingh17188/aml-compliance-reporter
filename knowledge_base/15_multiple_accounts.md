---
title: Multiple Accounts and Account Networks
category: Account Behaviour
risk_level: High
keywords:
  - multiple accounts
  - account network
  - mule accounts
  - linked accounts
  - many-to-one transfers
related_topics:
  - Structuring
  - Layering
  - Round-Tripping
---

# Multiple Accounts and Account Networks

## Overview

Multiple-account activity involves use of several accounts to send, receive, split, or pass funds. It may indicate account networks, mule activity, structuring, or layering.

## Why It Matters

Using multiple accounts can hide control relationships and make patterns harder to detect. It can also allow funds to be split and recombined.

## Typical Indicators

- Many senders transfer to one recipient.
- One sender transfers to many recipients.
- Similar amounts sent across multiple accounts.
- Accounts show pass-through activity.
- Newly created accounts involved in repeated transfers.

## Analyst Considerations

Review whether accounts are linked by owner, device, address, bank, timing, counterparty, or amount patterns. Consider mule or pass-through behaviour.

## Example Scenario

Several accounts receive similar deposits and quickly transfer funds to the same external beneficiary.

## Suggested Action

Escalate when multiple-account activity appears coordinated, repetitive, or lacks legitimate purpose.
