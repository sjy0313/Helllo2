package ch07_.inheritance.sec10_abstract.exam01;

public class PhoneExample {
	public static void main(String[] args) {
		//Phone phone = new Phone();
		//Cannot instantiate the type Phone
		// 즉, 추상클래스는 인스턴스화 불가
		SmartPhone smartPhone = new SmartPhone("홍길동");

		smartPhone.turnOn();
		smartPhone.internetSearch();
		smartPhone.turnOff();
	}
}