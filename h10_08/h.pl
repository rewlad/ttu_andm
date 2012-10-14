use strict;

sub for_pairs_from_file{
    my($filename,$fcb)=@_;
    open FF,'<',$filename or die;
    my $line;
    $line=~/^(\d+),(\d+)\s*$/ and &$fcb($1,$2) while $line = <FF>;
    close FF or die;
}

sub make_index{
    my(%ub,%bu);
    for_pairs_from_file('data.txt', sub{
        my($user,$book)=@_;
        $ub{$user}{$book} = $bu{$book} ||={};
        $bu{$book}{$user} = $ub{$user} ||={};
    });
    warn 'index_done';
    return \%bu;
}

sub mode_1{
    my $bu = make_index();
    for_pairs_from_file('book-pairs.txt', sub{
        my($book1,$book2) = @_;
        my $b1 = $$bu{$book1} || {};
        my $cnt = scalar grep{$$_{$book2}} values %$b1;
        print "for books $book1 and $book2 pair found $cnt common buyers\n";
    });
}

sub book_pairs{
    my($b1)=@_;
    my %bpair;
    $bpair{$_}++ for map{keys %$_} values %$b1;
    return \%bpair;
}

sub mode_2{
    my $bu = make_index();
    open FO,'>',"u2.out" or die;
    for my $book1 (keys %$bu){
        my $bpair = book_pairs($$bu{$book1} || die);
        my $cnt = $$bpair{$book1};
        for my $book2(keys %$bpair){
            $book1 lt $book2 or next;
            print FO "$book1,$book2,$$bpair{$book2}\n" or die;
        }
    }
    close FO or die;
}

sub mode_3{
    my $bu = make_index();
    open FO,'>',"u3.out" or die;
    for my $book1 (keys %$bu){
        my $bpair = book_pairs($$bu{$book1} || die);
        my $cnt = $$bpair{$book1};
        for my $book2(keys %$bpair){
            $book1 ne $book2 or next;
            my $p = int 100*$$bpair{$book2}/$cnt;
            print FO "$book1,$book2,$p%\n" or die;
        }
    }
    close FO or die;
}

sub mode_0{ print "Usage according to 'ülesanned': $0 1|2|3\n" }

main->$_() for "mode_".($ARGV[0]||'0');
