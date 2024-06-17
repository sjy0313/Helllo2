package ch08_interface.sec04_abstract_method_excerise;
// interface 로 객체를 구현할 떄는 implements
// 인터페이스끼리 상속을 할 때는 클래스와 마찬가지로 extends 키워드를 사용
public  class RemoteControlImpl implements RemoteControl {
		private String device; 		// remoteControl 한 대상 
		private int volume;			// 볼륨
		
		public RemoteControlImpl() {
			this.device = "장치"; // 사용자가 이름을 지정해주지 않으면 장치로 사용
		}
		//
		public RemoteControlImpl(String device) {
			this.device = device;
		}
		
		@Override
		public void turnOn() {
			System.out.printf("%s를 켭니다.\n", this.device);
		}
		
		@Override
		public void turnoff() {
			System.out.printf("%s를 끕니다.\n", this.device);
		}

		@Override
		public void setVolume(int volume) {
			if(volume>RemoteControl.MAX_VOLUME) {
				this.volume = RemoteControl.MAX_VOLUME;
				
			} else if(volume<RemoteControl.MIN_VOLUME) {
				this.volume = RemoteControl.MIN_VOLUME;
				
			} 
			else {
				this.volume = volume;
			}
			System.out.println("현재 Audio 볼륨: " + volume);
		}
}

