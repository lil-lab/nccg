#!/bin/bash
if [$1 == ""]; then
	echo 'Missing argument'
	exit
fi
for i in $(cat deploy.properties); do
	echo 'Deleting '$1/$i
	rm -rf $1/$i
	echo 'Copying '$i
	cp -rf $i $1
done
