package ch07_.inheritance.sec04_method_overiding.exam02_airplane;

public class SupersonicAirplaneExample {
	public static void main(String[] args) {
		SupersonicAirplane sa = new SupersonicAirplane();
		sa.takeOff();
		sa.fly();
		sa.flyMode = SupersonicAirplane.SUPERSONIC;
		sa.fly();
		sa.flyMode = SupersonicAirplane.NORMAL;
		sa.fly();
		sa.land();
	}
}
// 일반 비행모드일 경우 airplane의 fly()활용
//초음속 비행모드일 경우 오버라이딩 된 SUPERSONIC의 fly()활용 -> super.fly();