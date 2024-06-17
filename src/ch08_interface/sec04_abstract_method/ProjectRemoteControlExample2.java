package ch08_interface.sec04_abstract_method;

public class ProjectRemoteControlExample2 {
	public static void main(String[] args) {

		//Television 객체를 생성하고 인터페이스 변수에 대입
		RemoteControl rc = new Television();
		rc.turnOn();
		rc.setVolume(99);
		rc.turnOff();
		
	}
}