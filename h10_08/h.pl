use strict;

sub make_index{
  open FF,'<','data.txt' or die;
  my(%ub,%bu);
  /^(\d+),(\d+)\s*$/ ? do{ 
    $ub{$1}{$2} = $bu{$2} ||={}; 
    $bu{$2}{$1} = $ub{$1} ||={}; 
  } : 0 while <FF>;
  return \%bu;
}
print "read done\n";

my($mode,$book1,$book2) = \@ARGV;

sub mode_1{
    my($pkg,$mode,$book1,$book2)=@_;
    my $b1 = make_index()->{$book1} || {};
    my $cnt = scalar grep{$$_{$book2}} values %$b1;
    print "found: $cnt\n";
}

sub mode_2{
    my $bu = make_index();
    print "index done\n";
    open FO,'>',"data-pairs.out" or die;
    for my $b1_id (keys %$bu){
        my %bpair;
        $bpair{$_}++ for map{keys %$_} values %{$$bu{$b1_id}};
        print FO "$b1_id $_ $bpair{$_}\n" or die for keys %bpair;
    }
    close FO or die;
}

sub mode_3{

}

sub mode_{

}

main->$_(@ARGV) for "mode_".($ARGV[0]||'0');
