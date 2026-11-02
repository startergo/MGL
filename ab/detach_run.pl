#!/usr/bin/perl
# detach_run.pl <logfile> <command...> — double-fork + setsid then exec, so a
# long CTS run survives the invoking shell's process-group teardown (macOS has
# no setsid(1)).  Usage: perl ab/detach_run.pl /tmp/x.log nice -n 10 bash ab/...
use strict; use warnings; use POSIX qw(setsid);
my $log = shift @ARGV;
defined(my $pid = fork) or die "fork: $!";
exit 0 if $pid;                       # parent returns immediately
setsid() >= 0 or die "setsid: $!";
open(STDOUT, ">>", $log) or die "open $log: $!";
open(STDERR, ">>&", \*STDOUT);
open(STDIN, "<", "/dev/null");
exec @ARGV or die "exec @ARGV: $!";
