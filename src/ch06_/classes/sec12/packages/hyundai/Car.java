package ch06_.classes.sec12.packages.hyundai;

import ch06_.classes.sec12.packages.hankook.SnowTire;
import ch06_.classes.sec12.packages.kumho.AllSeasonTire;
	
public class Car {
	//부품 필드 선언
	ch06_.classes.sec12.packages.hankook.Tire tire1 = new ch06_.classes.sec12.packages.hankook.Tire();
	ch06_.classes.sec12.packages.kumho.Tire tire2 = new ch06_.classes.sec12.packages.kumho.Tire();
	
	// import 에 선언된 패키지는 패키지를 지정하지 않아도 된다 
	
	SnowTire tire3 = new SnowTire();
	AllSeasonTire tire4 = new AllSeasonTire();
}