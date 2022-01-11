x=60
while [ $x -le 100 ]
do
  echo "Welcome $x times"
  x=$(( $x + 5 ))
done