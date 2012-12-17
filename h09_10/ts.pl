use strict;
use utf8;
my $prev_c;
binmode STDIN,":utf8";
binmode STDOUT,":utf8";
for(<>){
    /^(\d+),"([^"]+)",,/ or print STDERR "undef:$_" and next;
    my($cur_c,$txt)=($1,$2);
    $txt=~s/\W/_/g;
    print $cur_c eq $prev_c ? " ": "\n";
    $prev_c = $cur_c;
    print lc $txt;
}