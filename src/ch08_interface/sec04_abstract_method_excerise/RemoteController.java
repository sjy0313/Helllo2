package ch08_interface.sec04_abstract_method_excerise;

public class RemoteController {
	public static void main(String[] args) {
		controller(new Audio());
		//Audio 객체를 생성하고 인터페이스 변수에 대입
		controller(new Television());
		//Television 객체를 생성하고 인터페이스 변수에 
		controller(new Television());
		controller(new Projection());
		
	}
	
	static void controller(RemoteControl rc) {
		rc.turnOn();
		rc.setVolume(3);
		rc.turnOff();
	}
	
}
